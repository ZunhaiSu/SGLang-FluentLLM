from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from functools import cache
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_free_port, get_ip, get_local_ip_by_remote

logger = logging.getLogger(__name__)

# Global debug tracking for decode_prefix_len transmission
DEBUG_PREFIX_TRACKER = {}

def debug_prefix_log(operation: str, request_id: str, prefix_len: int, additional_info: str = ""):
    """Debug utility to track decode_prefix_len transmission"""
    global DEBUG_PREFIX_TRACKER
    timestamp = time.time()
    DEBUG_PREFIX_TRACKER[request_id] = {
        "timestamp": timestamp,
        "operation": operation,
        "prefix_len": prefix_len,
        "additional_info": additional_info
    }
    logger.debug(f"[PREFIX_DEBUG] {operation}: req={request_id}, prefix_len={prefix_len}, info={additional_info}")


class CommonKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        draft_is_mla_backend: Optional[bool] = False,
    ):
        self.kv_args = args
        self.is_mla_backend = is_mla_backend
        self.draft_is_mla_backend = draft_is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        # for p/d multi node infer
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        self.world_size = server_args.world_size
        self.dp_size = server_args.dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        if not server_args.enable_dp_attention and server_args.dp_size != 1:
            raise ValueError(
                "If dp_attention is not enabled, dp size must be 1 in disaggregation mode."
            )

        self.rank_port = get_free_port()
        # Store decode prefix lengths received from decode side
        self.decode_prefix_lengths = {}
        
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._register_to_bootstrap()
            # Start bootstrap thread to receive decode prefix info from decode side
            self._start_bootstrap_thread()
            # For decode side receivers that need to fetch prefill info  
            self.prefill_tp_size_table: Dict[str, int] = {}
            self.prefill_dp_size_table: Dict[str, int] = {}
            # Also add connection_pool for compatibility
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.prefill_tp_size_table: Dict[str, int] = {}
            self.prefill_dp_size_table: Dict[str, int] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        if self.dist_init_addr:
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0])
        else:
            ip_address = get_ip()

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Prefill",
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.kv_args.engine_rank,
        }

        try:
            response = requests.put(url, json=payload)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill Failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(f"Prefill Failed to register to bootstrap server: {e}")

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def receive_decode_prefix_info(self, bootstrap_room: int) -> int:
        """Receive decode prefix info from decode side"""
        return self.decode_prefix_lengths.get(bootstrap_room, 0)
    
    def _start_bootstrap_thread(self):
        """Start thread to receive decode prefix info from decode side"""
        def bootstrap_thread():
            """This thread receives decode prefix info from decode engine"""
            server_socket = zmq.Context().socket(zmq.PULL)
            server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")
            
            logger.info(f"Bootstrap thread started on port {self.rank_port}, waiting for decode prefix info...")
            
            while True:
                try:
                    # Receive message with decode_prefix_len
                    msg_parts = server_socket.recv_multipart()
                    if len(msg_parts) >= 6:
                        bootstrap_room = int(msg_parts[0].decode("ascii"))
                        # decode_prefix_len is msg_parts[5]
                        decode_prefix_len = int(msg_parts[5].decode("ascii"))
                        
                        # Store decode prefix length
                        self.decode_prefix_lengths[bootstrap_room] = decode_prefix_len
                        logger.debug(f"Received decode_prefix_len={decode_prefix_len} for bootstrap_room={bootstrap_room}")
                    else:
                        logger.warning(f"Received unexpected message format with {len(msg_parts)} parts")
                except Exception as e:
                    logger.error(f"Error in bootstrap thread: {e}")
                    time.sleep(0.1)  # Avoid busy loop
        
        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def send_decode_prefix_info(self, bootstrap_room: int, decode_prefix_len: int):
        """Send decode prefix info to prefill side - in common backend this is handled via ZMQ thread"""
        # In common implementation, decode prefix info is sent via ZMQ in the bootstrap thread
        # Store it for the bootstrap thread to send
        self.decode_prefix_lengths[bootstrap_room] = decode_prefix_len
        logger.debug(f"Stored decode_prefix_len={decode_prefix_len} for bootstrap_room={bootstrap_room}")

    def update_status(self, bootstrap_room: int, status: KVPoll):
        """Update status for a bootstrap room - placeholder for compatibility"""
        # In common implementation, we don't need to track status
        pass


class CommonKVReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr

        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            self.prefill_tp_size, self.prefill_dp_size = self._get_prefill_dp_size_from_server()
                
            if self.prefill_tp_size is None or self.prefill_dp_size is None:
                logger.error(
                    f"Could not fetch prefill parallel info for bootstrap_addr: {self.bootstrap_addr}"
                )
            else:
                self.kv_mgr.prefill_tp_size_table[self.bootstrap_addr] = (
                    self.prefill_tp_size
                )
                self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = (
                    self.prefill_dp_size
                )
        else:
            self.prefill_tp_size = self.kv_mgr.prefill_tp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_dp_size = self.kv_mgr.prefill_dp_size_table[
                self.bootstrap_addr
            ]

        # Currently, we don't allow prefill instance and decode instance to
        # have different TP sizes per DP rank, except for models using MLA.
        if self.prefill_tp_size is not None and self.prefill_dp_size is not None:
            local_tp_size_per_dp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
            prefill_tp_size_per_dp_rank = self.prefill_tp_size // self.prefill_dp_size
        else:
            local_tp_size_per_dp_rank = 1  # fallback
            prefill_tp_size_per_dp_rank = 1

        if not self.kv_mgr.draft_is_mla_backend:
            assert (
                local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

        if local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            )
            self.required_dst_info_num = 1
            self.target_tp_ranks = [self.target_tp_rank]
        elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
            ) // (local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank)
            self.required_dst_info_num = (
                local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
            )
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            assert (
                self.kv_mgr.is_mla_backend
            ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks for non MLA models;
            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                    (self.kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank + 1)
                    * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                )
            ]

            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1

        self.target_dp_group = bootstrap_room % self.prefill_dp_size if self.prefill_dp_size is not None else 1

        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                bootstrap_info = self._get_bootstrap_info_from_server(
                    target_tp_rank,
                    self.target_dp_group,
                )
                if bootstrap_info is not None:
                    # NOTE: only support MLA for now: select one prefill rank as real rank
                    bootstrap_info["is_dummy"] = not bool(
                        target_tp_rank == self.target_tp_rank
                        or self.target_tp_rank is None
                    )
                    bootstrap_infos.append(bootstrap_info)
                else:
                    logger.error(
                        f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group}"
                    )
            self.bootstrap_infos = bootstrap_infos

            if len(self.bootstrap_infos) == 0:
                logger.error(
                    f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank}"
                )
            else:
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos
                # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
                self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0

    def _get_bootstrap_info_from_server(self, engine_rank, target_dp_group):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
            response = requests.get(url)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _get_prefill_dp_size_from_server(self) -> int:
        """Fetch the prefill parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return int(prefill_parallel_info["prefill_tp_size"]), int(
                    prefill_parallel_info["prefill_dp_size"]
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    def init(self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None, decode_prefix_len: Optional[int] = 0):
        """
        Notify prefill server about the kv indices and aux index,
        and send decode prefix length to prefill side
        """
        self.decode_prefix_len = decode_prefix_len
        
        if not self.bootstrap_infos:
            logger.error(f"No bootstrap infos available for room {self.bootstrap_room}")
            raise RuntimeError(f"No bootstrap infos available for room {self.bootstrap_room}")
        
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            is_dummy = bootstrap_info["is_dummy"]
            
            logger.debug(
                f"CommonKVReceiver init to {self.prefill_server_url} with bootstrap room {self.bootstrap_room}, "
                f"decode_prefix_len={decode_prefix_len}, is_dummy={is_dummy}"
            )
            
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                # Send decode_prefix_len as additional message part
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        str(decode_prefix_len).encode("ascii") if not is_dummy else b"",
                    ]
                )

    def poll(self) -> KVPoll:
        """In common implementation, return success immediately after init"""
        # For common backend, we assume transfer is always successful
        return KVPoll.Success

    def _register_kv_args(self):
        pass

    def failure_exception(self):
        raise Exception("Common KVReceiver Exception")


class CommonKVSender(BaseKVSender):
    def __init__(
        self, mgr: CommonKVManager, bootstrap_addr: str, bootstrap_room: int
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr.update_status(bootstrap_room, KVPoll.Bootstrapping)
        self.aux_index = None
        self.init_time = None
        self.decode_prefix_len = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None, decode_prefix_len: Optional[int] = 0):
        """Receive decode prefix length from decode side"""
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        self.decode_prefix_len = decode_prefix_len
        self.init_time = time.time()
        
        # Get decode_prefix_len from manager if available
        manager_decode_prefix_len = self.kv_mgr.receive_decode_prefix_info(self.bootstrap_room)
        if manager_decode_prefix_len > 0:
            self.decode_prefix_len = manager_decode_prefix_len
            
        logger.info(f"CommonKVSender init with decode_prefix_len={self.decode_prefix_len} for room {self.bootstrap_room}")

    def send(self, kv_indices: npt.NDArray[np.int64], start_idx: Optional[int] = 0):
        """In common implementation, this is a no-op since we don't actually transfer"""
        logger.debug(f"CommonKVSender send with kv_indices: {kv_indices}")

    def poll(self) -> KVPoll:
        """In common implementation, immediately return success after init"""
        if self.init_time is None:
            return KVPoll.Bootstrapping
        else:
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Common KVSender Exception")


class CommonKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.world_size = None
        self.dp_size = None
        self.tp_size_per_dp_rank = None
        self.prefill_port_table: Dict[int, Dict[int, Dict[str, Union[str, int]]]] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)

    async def _handle_route(self, request: web.Request):
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        role = data["role"]
        world_size = data["world_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        engine_rank = int(data["engine_rank"])

        if self.world_size is None:
            self.world_size = world_size

        if self.dp_size is None:
            self.dp_size = dp_size

        tp_size_per_dp_rank = world_size // dp_size
        if self.tp_size_per_dp_rank == None:
            self.tp_size_per_dp_rank = tp_size_per_dp_rank

        # Add lock to make sure thread-safe
        if role == "Prefill":
            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}

            self.prefill_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                f"Register Prefill bootstrap: {engine_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            prefill_parallel_info = {
                "prefill_tp_size": self.world_size,
                "prefill_dp_size": self.dp_size,
            }
            return web.json_response(prefill_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_group)][
                int(engine_rank)
            ]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._runner = web.AppRunner(self.app)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: 
        # For common backend, we assume transfer is always successful
        return KVPoll.Success

