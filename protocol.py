import json
import socket
import struct
import time

import torch

DEFAULT_PORT = 8182
PROTOCOL_VERSION = 1
HEADER_LEN_FMT = ">Q"
HEADER_LEN_SIZE = struct.calcsize(HEADER_LEN_FMT)
SOCKET_TIMEOUT = 120
IDLE_TIMEOUT = 4 * 3600
DECODE_TIMEOUT = 4 * 3600
CONNECT_RETRIES = 3
CONNECT_BACKOFF = 1.0
RECV_CHUNK = 1 << 20
MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_BLOB_BYTES = 8 * 1024 * 1024 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

ALLOWED_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def log(msg):
    print(f"[RemoteVAE {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_transport_dtype(mode, ip):
    if mode == "fp16":
        return "torch.float16"
    if mode == "fp32":
        return None
    if ip in _LOCAL_HOSTS:
        return None
    return "torch.float16"


def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, RECV_CHUNK))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(sock, header, blob=b""):
    header_bytes = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(HEADER_LEN_FMT, len(header_bytes)) + header_bytes)
    if blob:
        sock.sendall(blob)


def recv_header(sock):
    header_len = struct.unpack(HEADER_LEN_FMT, _recv_exact(sock, HEADER_LEN_SIZE))[0]
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {header_len} bytes")
    return json.loads(_recv_exact(sock, header_len).decode("utf-8"))


def recv_blob(sock, header):
    blob_size = header.get("blob_size", 0)
    if blob_size > MAX_BLOB_BYTES:
        raise ValueError(f"Blob too large: {blob_size} bytes")
    return _recv_exact(sock, blob_size) if blob_size else b""


def recv_packet(sock):
    header = recv_header(sock)
    return header, recv_blob(sock, header)


def set_socket_opts(sock, timeout=IDLE_TIMEOUT):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, "TCP_KEEPIDLE"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 8)
    sock.settimeout(timeout)


def pack_tensors(tensors, transport_dtype=None):
    meta = {}
    blobs = []
    offset = 0
    for name, t in tensors.items():
        if t is None:
            raise ValueError(f"Tensor '{name}' is None")
        t = t.detach().cpu()
        orig_dtype = str(t.dtype)
        if t.is_floating_point():
            if transport_dtype is not None:
                t = t.to(transport_dtype)
            elif t.dtype not in (torch.float16, torch.float32):
                t = t.to(torch.float32)
        t = t.contiguous()
        raw = t.numpy().tobytes()
        meta[name] = {
            "dtype": str(t.dtype),
            "orig_dtype": orig_dtype,
            "shape": list(t.shape),
            "offset": offset,
            "size": len(raw),
        }
        offset += len(raw)
        blobs.append(raw)
    return meta, b"".join(blobs)


def unpack_tensors(meta, blob):
    out = {}
    view = memoryview(blob)
    for name, info in meta.items():
        dtype_name = info["dtype"]
        if dtype_name not in ALLOWED_DTYPES:
            raise ValueError(f"Refusing to decode disallowed dtype: {dtype_name}")
        dtype = ALLOWED_DTYPES[dtype_name]
        start = info["offset"]
        raw = bytearray(view[start:start + info["size"]])
        t = torch.frombuffer(raw, dtype=dtype).reshape(info["shape"]).clone()
        orig = info.get("orig_dtype")
        if orig in ALLOWED_DTYPES and ALLOWED_DTYPES[orig] != dtype:
            t = t.to(ALLOWED_DTYPES[orig])
        out[name] = t
    return out


def jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    raise TypeError(f"Cannot send {type(obj).__name__} over RemoteVAE")
