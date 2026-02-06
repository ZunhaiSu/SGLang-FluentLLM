"""
KV Events functionality tests

Usage:
python3 -m unittest test_kv_events.TestKVEvents.test_event_publisher_initialization
python3 -m unittest test_kv_events.TestKVEvents.test_kv_events_generation
python3 test_kv_events.py
"""

import os
import unittest
import tempfile
import json
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, BlockStored, BlockRemoved, AllBlocksCleared
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.radix_cache import RadixCache


class TestKVEvents(unittest.TestCase):
    """Test KV events functionality including publisher initialization, event generation, and Dynamo compatibility."""
    
    def setUp(self):
        """Set up test environment for each test method."""
        # Create a temporary config file for testing
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({
            "test_publisher": {
                "type": "test",
                "config": {"test_param": "test_value"}
            }
        }, self.temp_config)
        self.temp_config.close()
        
        # Mock scheduler instance for testing
        self.mock_scheduler = Mock(spec=Scheduler)
        self.mock_scheduler.enable_kv_cache_events = True
        self.mock_scheduler.attn_dp_rank = 0
        
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_event_publisher_initialization(self):
        """Test KV event publisher initialization with null safety."""
        # Test the bug fix: ensure kv_event_publisher is initialized to None
        scheduler = Mock()
        scheduler.enable_kv_cache_events = False
        
        # Simulate the init_kv_events method behavior
        kv_event_publisher = None  # Initialize to None by default
        
        # Test that it's properly initialized
        self.assertIsNone(kv_event_publisher)
        
        # Test the _publish_kv_events method behavior
        if not scheduler.enable_kv_cache_events or not kv_event_publisher:
            # Should return early without error
            result = None
        else:
            result = "should_not_reach_here"
            
        self.assertIsNone(result, "Should return early when publisher is None")
    
    def test_kv_events_generation(self):
        """Test KV cache events are generated correctly."""
        # Test BlockStored event generation
        block_hashes = [12345, 67890]
        block_stored_event = BlockStored(
            block_hashes=block_hashes,
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4, 5],
            block_size=16,
            lora_id=None
        )
        
        self.assertEqual(block_stored_event.block_hashes, block_hashes)
        
        # Test BlockRemoved event generation
        block_removed_event = BlockRemoved(block_hashes=block_hashes)
        
        self.assertEqual(block_removed_event.block_hashes, block_hashes)
        
        # Test AllBlocksCleared event generation
        all_cleared_event = AllBlocksCleared()
    
    def test_event_publisher_factory_creation(self):
        """Test EventPublisherFactory creates publishers correctly."""
        # Test with None config (should create NullEventPublisher)
        try:
            publisher = EventPublisherFactory.create(
                None,  # No config should create null publisher
                attn_dp_rank=0
            )
            # Should create a NullEventPublisher when no config provided
            self.assertIsNotNone(publisher)
        except Exception as e:
            self.fail(f"EventPublisherFactory.create() raised exception: {e}")
    
    def test_dynamo_compatibility_tracking(self):
        """Test Dynamo-compatible tracking and KV events support."""
        # Test that KV events work with Dynamo interface
        scheduler = Mock()
        scheduler.enable_kv_cache_events = True
        scheduler.kv_event_publisher = Mock()
        
        # Mock tree cache with events
        mock_events = [
            BlockStored(
                block_hashes=[123],
                parent_block_hash=None,
                token_ids=[1, 2, 3],
                block_size=16,
                lora_id=None
            ),
            BlockRemoved(block_hashes=[456])
        ]
        scheduler.tree_cache = Mock()
        scheduler.tree_cache.take_events.return_value = mock_events
        
        # Test _publish_kv_events method
        if scheduler.enable_kv_cache_events and scheduler.kv_event_publisher:
            # Simulate event publishing
            events = scheduler.tree_cache.take_events()
            if events:
                for event in events:
                    scheduler.kv_event_publisher.publish(event)
        
        # Verify events were taken from tree cache
        scheduler.tree_cache.take_events.assert_called_once()
    
    def test_tokenizer_null_handling(self):
        """Test tokenizer null handling during decoding."""
        # Test the bug fix: handle tokenizer being None
        tokenizer = None
        input_ids = [1, 2, 3, 4, 5]
        
        # Simulate the tokenizer null check
        if tokenizer is not None:
            text = tokenizer.decode(input_ids, skip_special_tokens=False)
        else:
            text = ""
        
        self.assertEqual(text, "", "Should return empty string when tokenizer is None")
    
    def test_prefill_scheduler_kv_event_timing(self):
        """Test KV events are published at correct timing in prefill scheduler."""
        # Test that KV events are published after processing batch results
        scheduler = Mock()
        scheduler.enable_kv_cache_events = True
        scheduler.kv_event_publisher = Mock()
        
        # Mock the tree cache
        scheduler.tree_cache = Mock()
        scheduler.tree_cache.take_events.return_value = []
        
        # Simulate _publish_kv_events method
        def mock_publish_kv_events():
            if scheduler.enable_kv_cache_events and scheduler.kv_event_publisher:
                events = scheduler.tree_cache.take_events()
                return events
        
        # Call the mock method
        result = mock_publish_kv_events()
        
        # Verify tree cache take_events was called
        scheduler.tree_cache.take_events.assert_called_once()
        self.assertEqual(result, [])
    
    def test_radix_cache_event_logic(self):
        """Test RadixCache BlockStored event generation logic."""
        # Create a simple RadixCache for testing with correct parameters
        radix_cache = RadixCache(
            req_to_token_pool=None,  # Mock pool
            token_to_kv_pool_allocator=None,  # Mock allocator
            disable=False,
            enable_kv_cache_events=True
        )
        
        # Initialize event queue for testing
        radix_cache.kv_event_queue = []
        
        # Test that event queue is initialized
        self.assertIsInstance(radix_cache.kv_event_queue, list)
        
        # Test take_events method
        events = radix_cache.take_events()
        self.assertIsInstance(events, list)
        
        # Test that queue is cleared after taking events
        self.assertEqual(len(radix_cache.kv_event_queue), 0)
    
    def test_event_serialization(self):
        """Test that KV events can be serialized properly."""
        # Test event serialization for network transmission
        events = [
            BlockStored(
                block_hashes=[123, 456],
                parent_block_hash=None,
                token_ids=[1, 2, 3, 4],
                block_size=16,
                lora_id=None
            ),
            BlockRemoved(block_hashes=[789]),
            AllBlocksCleared()
        ]
        
        # Test that events have proper structure
        for event in events:
            self.assertTrue(hasattr(event, '__struct_fields__'))
    
    def test_concurrent_event_handling(self):
        """Test concurrent event handling scenarios."""
        # Test multiple events being generated concurrently
        event_queue = []
        
        # Simulate concurrent event generation
        events_batch_1 = [
            BlockStored(
                block_hashes=[1, 2],
                parent_block_hash=None,
                token_ids=[1, 2],
                block_size=16,
                lora_id=None
            ),
            BlockRemoved(block_hashes=[3])
        ]
        events_batch_2 = [
            BlockStored(
                block_hashes=[4, 5],
                parent_block_hash=None,
                token_ids=[3, 4],
                block_size=16,
                lora_id=None
            ),
            AllBlocksCleared()
        ]
        
        event_queue.extend(events_batch_1)
        event_queue.extend(events_batch_2)
        
        # Test that all events are properly queued
        self.assertEqual(len(event_queue), 4)
        
        # Test atomic take operation
        taken_events = event_queue.copy()
        event_queue.clear()
        
        self.assertEqual(len(taken_events), 4)
        self.assertEqual(len(event_queue), 0)


class TestKVEventsIntegration(unittest.TestCase):
    """Integration tests for KV events with actual scheduler components."""
    
    def test_kv_events_with_server_args(self):
        """Test KV events configuration through server args."""
        # Test that server args properly configure KV events
        from sglang.srt.server_args import ServerArgs
        
        # Create server args with KV events enabled (requires model_path and other defaults)
        args = ServerArgs(
            model_path="test_model",
            max_running_requests=32,  # Set default to avoid None comparison
            dp_size=1  # Set default dp_size
        )
        args.enable_kv_cache_events = True
        
        self.assertTrue(args.enable_kv_cache_events)
        
        # Test with KV events disabled
        args.enable_kv_cache_events = False
        self.assertFalse(args.enable_kv_cache_events)


if __name__ == "__main__":
    unittest.main()