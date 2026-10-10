"""Persistent compilation cache: entries are written atomically, so a concurrent reader never decompresses a
truncated entry ("Error -5 while decompressing data"); the bytes written and the compiled results are unchanged."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest
import zlib
from unittest import mock

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax._src import compilation_cache, lru_cache  # noqa: E402

from experiments.exp12 import precision  # noqa: E402

ENTRIES = 30


def _reader(path, n, ready, out):
    """Poll every key; count reads that fail to decompress (a partially written entry)."""
    cache = lru_cache.LRUCache(path, max_size=-1)
    done, failures = set(), 0
    ready.set()
    deadline = time.time() + 120
    while len(done) < n and time.time() < deadline:
        for i in range(n):
            if i in done:
                continue
            val = cache.get(f"k{i}")
            if val is None:
                continue
            try:
                zlib.decompress(val)
                done.add(i)
            except zlib.error:
                failures += 1
    out.put((failures, len(done)))


def _original_put():
    put = lru_cache.LRUCache.put
    return getattr(put, "__wrapped__", put) if getattr(put, "atomic", False) else put


class AtomicCacheWriteTest(unittest.TestCase):
    def setUp(self):
        self.original = _original_put()
        lru_cache.LRUCache.put = self.original  # each test starts from jax's own put
        self.temp = tempfile.TemporaryDirectory()
        self.dir = self.temp.name

    def tearDown(self):
        lru_cache.LRUCache.put = self.original
        self.temp.cleanup()

    def test_concurrent_reader_never_sees_a_truncated_entry(self):
        precision.atomic_cache_writes()
        ctx = mp.get_context("spawn")
        ready, out = ctx.Event(), ctx.Queue()
        reader = ctx.Process(target=_reader, args=(self.dir, ENTRIES, ready, out))
        reader.start()
        cache = lru_cache.LRUCache(self.dir, max_size=-1)
        payload = zlib.compress(os.urandom(8 << 20))
        self.assertTrue(ready.wait(timeout=120), "reader process did not start")  # write only while it polls
        for i in range(ENTRIES):
            cache.put(f"k{i}", payload)
            time.sleep(0.01)
        failures, read = out.get(timeout=180)
        reader.join(timeout=30)
        self.assertEqual(reader.exitcode, 0)
        self.assertEqual(read, ENTRIES)
        self.assertEqual(failures, 0, "a reader decompressed a partially written cache entry")

    def test_empty_key_retains_jax_error(self):
        precision.atomic_cache_writes()
        cache = lru_cache.LRUCache(self.dir, max_size=-1)
        with self.assertRaisesRegex(ValueError, "key cannot be empty"):
            cache.put("", b"value")
        self.assertEqual(os.listdir(self.dir), [])

    def test_failed_publication_leaves_no_visible_or_temporary_entry(self):
        precision.atomic_cache_writes()
        cache = lru_cache.LRUCache(self.dir, max_size=-1)
        with mock.patch.object(precision.os, "replace", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                cache.put("key", b"complete")
        self.assertIsNone(cache.get("key"))
        self.assertEqual(os.listdir(self.dir), [])
        cache.put("key", b"complete")
        self.assertEqual(cache.get("key"), b"complete")

    def test_entry_is_invisible_until_publication(self):
        precision.atomic_cache_writes()
        cache = lru_cache.LRUCache(self.dir, max_size=-1)
        replace = os.replace
        payload = zlib.compress(os.urandom(4096))

        def publish(source, destination):
            self.assertIsNone(cache.get("key"))
            self.assertEqual(Path(source).read_bytes(), payload)
            replace(source, destination)

        with mock.patch.object(precision.os, "replace", side_effect=publish):
            cache.put("key", payload)
        self.assertEqual(cache.get("key"), payload)

    def test_concurrent_writers_publish_only_complete_payloads(self):
        precision.atomic_cache_writes()
        cache = lru_cache.LRUCache(self.dir, max_size=-1)
        payloads = [zlib.compress(os.urandom(1 << 20)) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda value: cache.put("key", value), payloads))
        self.assertIn(cache.get("key"), payloads)
        self.assertEqual(sorted(os.listdir(self.dir)), ["key-atime", "key-cache"])

    def test_entry_bytes_and_layout_unchanged(self):
        precision.atomic_cache_writes()
        cache = lru_cache.LRUCache(self.dir, max_size=-1)
        cache.put("key", b"first")
        cache.put("key", b"second")  # an existing entry is never replaced, as in jax's put
        self.assertEqual(cache.get("key"), b"first")
        names = sorted(os.listdir(self.dir))
        self.assertEqual(names, ["key-atime", "key-cache"])
        self.assertEqual(len((Path(self.dir) / "key-atime").read_bytes()), 8)

    @unittest.skipIf(lru_cache.filelock is None, "jax needs filelock for an evicting cache; not installed here")
    def test_eviction_cache_keeps_jax_put(self):
        precision.atomic_cache_writes()
        cache = lru_cache.LRUCache(self.dir, max_size=1 << 20)
        cache.put("key", b"value")
        self.assertEqual(cache.get("key"), b"value")

    def test_configure_installs_once_and_a_cached_executable_gives_identical_results(self):
        old = {k: os.environ.get(k) for k in (precision.CACHE_DIR_ENV,)}
        os.environ[precision.CACHE_DIR_ENV] = self.dir
        try:
            precision.configure_compilation_cache()
            installed = lru_cache.LRUCache.put
            precision.configure_compilation_cache()
            self.assertTrue(installed.atomic)
            self.assertIs(lru_cache.LRUCache.put, installed)
            f = jax.jit(lambda x: jnp.sin(x) @ x.T + 1.0)
            x = jnp.arange(64.0).reshape(8, 8) / 64
            first = f(x)
            self.assertTrue(any(n.endswith("-cache") for n in os.listdir(self.dir)))
            self.assertFalse(any(n.endswith(".tmp") for n in os.listdir(self.dir)))
            hits = precision._cache_events.get("/jax/compilation_cache/cache_hits", 0)
            jax.clear_caches()
            second = f(x)
            self.assertGreater(precision._cache_events.get("/jax/compilation_cache/cache_hits", 0), hits)
            first, second = np.asarray(first), np.asarray(second)
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.shape, second.shape)
            self.assertEqual(first.tobytes(), second.tobytes())
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            jax.config.update("jax_compilation_cache_dir", None)
            compilation_cache.reset_cache()


if __name__ == "__main__":
    unittest.main()
