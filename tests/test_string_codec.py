import unittest

from workrunner.string_codec import customDecoder, customEncoder


class StringCodecTests(unittest.TestCase):
    def test_short_frame_counts_bytes(self):
        self.assertEqual(customEncoder(""), b"\x00")
        self.assertEqual(customEncoder("café"), b"\x05caf\xc3\xa9")
        self.assertEqual(customDecoder(b"\x05caf\xc3\xa9"), "café")

    def test_length_boundary(self):
        for size in (254, 255, 256, 65536):
            with self.subTest(size=size):
                payload = b"x" * size
                header = bytes([size]) if size < 255 else b"\xff" + size.to_bytes(8, "little")
                self.assertEqual(customEncoder(payload.decode()), header + payload)
                self.assertEqual(customDecoder(header + payload), payload.decode())

    def test_round_trip(self):
        for text in ("", "line\nwith\0nul", "文🌱", "e\u0301", "\ud800"):
            with self.subTest(text=text):
                self.assertEqual(customDecoder(customEncoder(text)), text)

    def test_invalid_frames(self):
        for frame in (b"", b"\xff", b"\xff" + bytes(8), b"\x02x", b"\x00x", b"\x01\xff"):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                customDecoder(frame)
