import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


from workrunner.string_codec import customDecoder


backend = importlib.import_module("workrunner.stream_digest")
StreamDigest = backend.StreamDigest
VECTORS = json.loads((Path(__file__).parent / "fixtures" / "stream_digest_vectors.json").read_text())
CASES = {case["input_len"]: case for case in VECTORS["cases"]}
KEY = VECTORS["key"].encode("ascii")
CONTEXT = customDecoder(bytes.fromhex(VECTORS["context_frame"]))
MODES = (("hash", {}), ("keyed_hash", {"key": KEY}),
         ("derive_key", {"derive_key_context": CONTEXT}))

CONTEXT_OUTPUTS = (
    ("", "ee17756c11048d9d3690236f5615954a1384feee56c58e14da23e3da70c876dc3"
     "47369cbd1323e364c21d20a6b82366babd3ca2f67d5e8567d370a0d6660c6334e"),
    (customDecoder(bytes.fromhex('1cd8a2d8b2d985d8a7db8cd8b420434e4e5f33656b616c6220f09f9492')), "d3fd02fe3c387d38d8c41999e9970fc15a87855a29c4f9316c44db7f405e5af0"
     "1a87654fe51c3bb169cb37824a319bf60edebf20e57d508daf0b73aa9055ba3057"),
)
SEEK_OUTPUTS = (
    (2**32 + 63, {}, "a79023e7eb71e3715e861a8381c7340655b39093c873ac4976350dbaad6ce8cde"
     "b6a2e866ed062558ca0f619b8040e10b1ab092e0eca5a472055b92029003ba80d"),
    (2**32 + 63, {"key": KEY}, "c94187f1d956ab2620699f4e6b548632e8b0b045764e76fb99414faf26a0ed36e"
     "9b1e3522e084d5e2812866c3e4058a25e5bd7128c74e8e11c44d233b2e314da83"),
    (2**64 - 1, {}, "e8c1a42ee3a6b55337e844e5d1c986b7ae770b1568c163759fe85e509e6bf1988"
     "c2bbdaba564f2884d14dadeaa76b498bc332a481c296553eadc0399644f807a35"),
    (2**64 - 1, {"key": KEY}, "724b2ad87989d0c086bc0b40ca9f5ee2210adb72333e56a7162b7baa627fe64a0"
     "d8207040a680b7215fbdb1aea412a95845c8c3c3a49cf00c67aca1b4706346f48"),
)


def patterned(length):
    return bytes(index % 251 for index in range(length))


def expected(length, mode, output_length=32, seek=0):
    value = bytes.fromhex(CASES[length][mode])
    return value[seek:seek + output_length]


class BackendChecks:
    def test_official_vectors_and_seeked_extended_output(self):
        for length in CASES:
            for mode, options in MODES:
                with self.subTest(length=length, mode=mode):
                    hasher = StreamDigest(patterned(length), **options)
                    output = expected(length, mode, 131)
                    self.assertEqual(hasher.digest(), output[:32])
                    self.assertEqual(hasher.hexdigest(), output[:32].hex())
                    self.assertEqual(hasher.digest(131), output)
                    self.assertEqual(hasher.hexdigest(131), output.hex())
                    for start, size in ((0, 0), (31, 65), (63, 65), (64, 67), (127, 4)):
                        self.assertEqual(hasher.digest(size, seek=start), output[start:start + size])
                        self.assertEqual(hasher.hexdigest(size, seek=start), output[start:start + size].hex())

    def test_irregular_streaming_matches_official_vectors(self):
        data = patterned(8193)
        for mode, options in MODES:
            with self.subTest(mode=mode):
                hasher = StreamDigest(**options)
                position = 0
                sizes = (1, 63, 64, 65, 511, 1024, 1025)
                index = 0
                while position < len(data):
                    end = min(position + sizes[index % len(sizes)], len(data))
                    self.assertIs(hasher.update(memoryview(data)[position:end]), hasher)
                    self.assertIs(hasher.update(b""), hasher)
                    position = end
                    index += 1
                self.assertEqual(hasher.digest(131), expected(8193, mode, 131))

    def test_empty_and_unicode_derivation_contexts_match_fixed_outputs(self):
        data = patterned(1025)
        for context, output in CONTEXT_OUTPUTS:
            with self.subTest(context=context):
                hasher = StreamDigest(data[:64], derive_key_context=context)
                hasher.update(data[64:])
                self.assertEqual(hasher.hexdigest(65), output)

    def test_large_seek_offsets_match_fixed_outputs(self):
        for seek, options, output in SEEK_OUTPUTS:
            with self.subTest(seek=seek, keyed=bool(options)):
                hasher = StreamDigest(patterned(1025), **options)
                self.assertEqual(hasher.hexdigest(65, seek=seek), output)
                self.assertEqual(hasher.digest(65, seek=seek), bytes.fromhex(output))
                self.assertEqual(hasher.digest(0, seek=seek), b"")

    def test_digest_does_not_finalize_or_mutate_stream(self):
        data = patterned(1025)
        for mode, options in MODES:
            with self.subTest(mode=mode):
                hasher = StreamDigest(data[:64], **options)
                self.assertEqual(hasher.digest(131), expected(64, mode, 131))
                self.assertEqual(hasher.digest(), expected(64, mode))
                hasher.update(data[64:1024])
                self.assertEqual(hasher.digest(131), expected(1024, mode, 131))
                hasher.update(data[1024:])
                self.assertEqual(hasher.digest(131), expected(1025, mode, 131))

    def test_copy_owns_independent_chunk_and_tree_state(self):
        data = patterned(8193)
        for mode, options in MODES:
            for split in (63, 1023, 3072):
                with self.subTest(mode=mode, split=split):
                    original = StreamDigest(data[:split], **options)
                    cloned = original.copy()
                    self.assertIsNot(original, cloned)
                    cloned.update(data[split:7168])
                    original.update(data[split:])
                    self.assertEqual(cloned.digest(131), expected(7168, mode, 131))
                    self.assertEqual(original.digest(131), expected(8193, mode, 131))
                    cloned.reset()
                    self.assertEqual(original.digest(), expected(8193, mode))

    def test_reset_preserves_key_or_context(self):
        for mode, options in MODES:
            with self.subTest(mode=mode):
                hasher = StreamDigest(patterned(3073), **options)
                hasher.reset()
                self.assertEqual(hasher.digest(131), expected(0, mode, 131))
                hasher.update(patterned(65))
                self.assertEqual(hasher.digest(131), expected(65, mode, 131))

    def test_buffer_inputs_are_consumed_before_they_can_change(self):
        for constructor in (bytes, bytearray, memoryview):
            with self.subTest(buffer_type=constructor.__name__):
                data = bytearray(patterned(64))
                value = constructor(data)
                hasher = StreamDigest(value)
                if isinstance(value, bytearray):
                    value[:] = bytes(64)
                data[:] = bytes(64)
                self.assertEqual(hasher.digest(), expected(64, "hash"))
                tail = bytearray(patterned(65)[64:])
                value = constructor(tail)
                hasher.update(value)
                if isinstance(value, bytearray):
                    value[:] = bytes(len(value))
                tail[:] = bytes(len(tail))
                self.assertEqual(hasher.digest(), expected(65, "hash"))
        for constructor in (bytearray, memoryview):
            with self.subTest(key_type=constructor.__name__):
                self.assertEqual(StreamDigest(patterned(64), key=constructor(KEY)).digest(),
                                 expected(64, "keyed_hash"))

    def test_mutable_key_is_copied_for_updates_and_reset(self):
        data = patterned(65)
        for constructor in (bytearray, memoryview):
            with self.subTest(key_type=constructor.__name__):
                key = constructor(bytearray(KEY))
                hasher = StreamDigest(data[:64], key=key)
                key[:] = bytes(32)
                hasher.update(data[64:])
                self.assertEqual(hasher.digest(131), expected(65, "keyed_hash", 131))
                hasher.reset()
                hasher.update(data[:64])
                self.assertEqual(hasher.digest(131), expected(64, "keyed_hash", 131))

    def test_noncontiguous_input_and_key_buffers_are_rejected(self):
        noncontiguous = memoryview(bytes(64))[::2]
        with self.assertRaises(BufferError):
            StreamDigest(noncontiguous)
        with self.assertRaises(BufferError):
            StreamDigest().update(noncontiguous)
        with self.assertRaises(BufferError):
            StreamDigest(key=noncontiguous)

    def test_file_mapping_matches_streaming_and_preserves_prefix(self):
        data = patterned(3073)
        for mode, options in MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.bin"
                path.write_bytes(data[64:])
                hasher = StreamDigest(data[:64], **options)
                self.assertIs(hasher.update_mmap(path), hasher)
                self.assertEqual(hasher.digest(131), expected(3073, mode, 131))
                path.write_bytes(b"")
                self.assertIs(hasher.update_mmap(path), hasher)
                self.assertEqual(hasher.digest(), expected(3073, mode))
                self.assertEqual(StreamDigest(**options).update_mmap(path).digest(), expected(0, mode))
                with self.assertRaises(OSError):
                    hasher.update_mmap(Path(directory) / "missing.bin")

    def test_public_sizes_and_compatibility_options(self):
        self.assertEqual(StreamDigest.name.encode(), bytes.fromhex("626c616b6533"))
        for attribute, value in (("digest_size", 32), ("block_size", 64), ("key_size", 32), ("AUTO", -1)):
            self.assertEqual(getattr(StreamDigest, attribute), value)
            self.assertEqual(getattr(StreamDigest(), attribute), value)
        for threads in (1, 2, StreamDigest.AUTO):
            self.assertEqual(StreamDigest(patterned(64), max_threads=threads, usedforsecurity=False).digest(),
                             expected(64, "hash"))
        self.assertEqual(StreamDigest(patterned(64), key=None, derive_key_context=None).digest(),
                         expected(64, "hash"))
        self.assertEqual(StreamDigest(None).digest(), expected(0, "hash"))

    def test_invalid_keys_modes_and_buffers_are_rejected(self):
        for size in (0, 1, 31, 33, 64):
            with self.subTest(key_length=size), self.assertRaises((ValueError, TypeError)):
                StreamDigest(key=bytes(size))
        with self.assertRaises(ValueError):
            StreamDigest(key=KEY, derive_key_context=CONTEXT)
        for context in (42, b"context"):
            with self.subTest(context=context), self.assertRaises(TypeError):
                StreamDigest(derive_key_context=context)
        for value in ("text", 42, []):
            with self.subTest(buffer=value), self.assertRaises(TypeError):
                StreamDigest(value)
            with self.subTest(update_buffer=value), self.assertRaises(TypeError):
                StreamDigest().update(value)
        with self.assertRaises(TypeError):
            StreamDigest().update(None)
        for threads in (0, -2):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                StreamDigest(max_threads=threads)

    def test_invalid_output_ranges_are_rejected(self):
        for method_name in ("digest", "hexdigest"):
            method = getattr(StreamDigest(), method_name)
            for options in ({"length": -1}, {"length": 1.5}, {"seek": -1},
                            {"seek": 1.5}, {"seek": 1 << 64}):
                with self.subTest(method=method_name, options=options), \
                        self.assertRaises((TypeError, ValueError, OverflowError)):
                    method(**options)


class AutomaticBackendTests(BackendChecks, unittest.TestCase):
    pass


class PythonFallbackTests(BackendChecks, unittest.TestCase):
    def setUp(self):
        self.fallback = patch.object(backend, "_native", return_value=None)
        self.fallback.start()
        self.addCleanup(self.fallback.stop)


class TreeWrapperTests(unittest.TestCase):
    def test_compression_matches_an_official_single_block_root(self):
        from workrunner import hashing

        iv = bytes.fromhex("67e6096a85ae67bb72f36e3c3af54fa5"
                           "7f520e518c68059babd9831f19cde05b")
        output = expected(64, "hash", 64)
        self.assertEqual(hashing.compress(patterned(64), iv, 0, 64, 11), output)
        with patch.object(hashing, "_native", return_value=None):
            self.assertEqual(hashing.compress(patterned(64), iv, 0, 64, 11), output)

    def test_invalid_tree_arguments_never_reach_native_backend(self):
        from workrunner import hashing

        block, cv = bytes(64), bytes(32)
        invalid = [
            (hashing.compress, (bytes(size), cv, 0, 64, 11)) for size in (0, 63, 65)
        ] + [
            (hashing.compress, (block, bytes(size), 0, 64, 11)) for size in (0, 31, 33)
        ] + [
            (hashing.compress, (block, cv, value, 64, 11)) for value in (-1, 1 << 64, 1.5)
        ] + [
            (hashing.compress, (block, cv, 0, value, 11)) for value in (-1, 65, 1 << 32, 1.5)
        ] + [
            (hashing.compress, (block, cv, 0, 64, value)) for value in (-1, 1 << 32, 1.5)
        ] + [
            (hashing.chunk_cv, (bytes(1025), cv)),
            (hashing.chunk_cv, (b"data", bytes(31))),
            (hashing.chunk_cv, (b"data", bytes(33))),
            (hashing.chunk_cv, (b"data", cv, -1)),
            (hashing.chunk_cv, (b"data", cv, 1 << 64)),
            (hashing.parent_cv, (bytes(31), cv, cv)),
            (hashing.parent_cv, (cv, bytes(33), cv)),
            (hashing.parent_cv, (cv, cv, b"")),
        ]
        with patch.object(hashing, "_native") as native:
            for function, arguments in invalid:
                with self.subTest(function=function.__name__, arguments=arguments), \
                        self.assertRaises((ValueError, TypeError, OverflowError, struct.error)):
                    function(*arguments)
            native.assert_not_called()

    def test_python_tree_matches_fixed_three_leaf_root_and_proof(self):
        from workrunner import hashing, primitives

        with patch.object(hashing, "_native", return_value=None):
            tree = primitives.DigestTree(patterned(3072), KEY)
            self.assertEqual(tree.root, expected(3072, "keyed_hash"))
            proof = tree.get_multileaf_proof([0, 2])
            self.assertTrue(proof.verify(KEY))
            self.assertFalse(proof.verify(bytes(32)))
            first = proof.leaf_data[0]
            proof.leaf_data[0] = bytes([first[0] ^ 1]) + first[1:]
            self.assertFalse(proof.verify(KEY))


class DependencyIsolationTests(unittest.TestCase):
    def test_backend_imports_and_hashes_when_external_package_is_blocked(self):
        source = Path(backend.__file__).resolve().parent.parent
        script = '''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
from workrunner.string_codec import customDecoder
external_package = customDecoder(bytes.fromhex("06626c616b6533"))
class DenyReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == external_package:
            raise AssertionError('external hashing package was imported')
sys.meta_path.insert(0, DenyReference())
from workrunner.stream_digest import StreamDigest
assert StreamDigest().hexdigest() == sys.argv[2]
assert external_package not in sys.modules
'''
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(source),
                                 expected(0, "hash").hex()],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
