def customEncoder(text: str) -> bytes:
    payload = text.encode("utf-8", errors="surrogatepass")
    size = len(payload)
    if size < 255:
        return bytes([size]) + payload
    return b"\xff" + size.to_bytes(8, "little") + payload


def customDecoder(frame: bytes) -> str:
    if not frame:
        raise ValueError("Missing string frame header")
    size = frame[0]
    start = 1
    if size == 255:
        if len(frame) < 9:
            raise ValueError("Incomplete string frame header")
        size = int.from_bytes(frame[1:9], "little")
        if size < 255:
            raise ValueError("Unexpected extended string frame header")
        start = 9
    if len(frame) - start != size:
        raise ValueError("String frame length mismatch")
    return frame[start:].decode("utf-8", errors="surrogatepass")
