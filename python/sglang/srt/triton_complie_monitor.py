"""
Can be used to monitor the number of triton kernel compilations at runtime.
"""

import triton
from collections import defaultdict
import threading

class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def inc(self):
        with self._lock:
            self._value += 1

    @property
    def value(self):
        with self._lock:
            return self._value

class TritonCompileMonitor:
    def __init__(self):
        self.counts = defaultdict(AtomicCounter)
        self.original_compile = triton.compiler.compile

    def compile_with_counter(self, src, *args, **kwargs):
        kernel_name = src.name
        self.counts[kernel_name].inc()
        print(f"Compiling {kernel_name} (total: {self.counts[kernel_name].value})")
        return self.original_compile(src, *args, **kwargs)

    def install(self):
        triton.compiler.compile = self.compile_with_counter

    def get_counts(self, kernel_name=None):
        if kernel_name:
            return self.counts[kernel_name].value
        return {k: v.value for k, v in self.counts.items()}
