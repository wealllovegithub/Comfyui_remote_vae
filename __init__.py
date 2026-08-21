import gc
import hmac
import os
import socket
import struct
import threading
import time

import torch

import comfy.model_management
from .protocol import (
    ALLOWED_DTYPES,
    CONNECT_BACKOFF,
    CONNECT_RETRIES,
    DECODE_TIMEOUT,
    DEFAULT_PORT,
    IDLE_TIMEOUT,
    PROTOCOL_VERSION,
    SOCKET_TIMEOUT,
    jsonable,
    log,
    pack_tensors,
    recv_blob,
    recv_header,
    recv_packet,
    resolve_transport_dtype,
    send_packet,
    set_socket_opts,
    unpack_tensors,
)


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _empty_device_cache():
    gc.collect()
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    _malloc_trim()


def _offload_vae_obj(vae):
    """Move this VAE off GPU. Weights stay on CPU so the next encode/decode can reload."""
    patcher = getattr(vae, "patcher", None)
    if patcher is not None:
        try:
            comfy.model_management.unload_model_and_clones(patcher, unload_additional_models=True)
        except Exception as exc:
            log(f"VAE GPU unload: {exc}")
            try:
                patcher.unpatch_model()
            except Exception:
                pass
            model = getattr(patcher, "model", None) or getattr(vae, "first_stage_model", None)
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
    else:
        model = getattr(vae, "first_stage_model", None)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
    _empty_device_cache()


_CLEANUP_FNS_ATTR = "_remote_av_cleanup_fns"


def _install_master_cleanup_hook(flush_fn):
    try:
        import execution
    except Exception:
        return
    fns = getattr(execution.PromptExecutor, _CLEANUP_FNS_ATTR, None)
    if fns is None:
        fns = []
        setattr(execution.PromptExecutor, _CLEANUP_FNS_ATTR, fns)
        orig = execution.PromptExecutor.execute

        def wrapped(self, *args, **kwargs):
            try:
                return orig(self, *args, **kwargs)
            finally:
                for fn in list(getattr(execution.PromptExecutor, _CLEANUP_FNS_ATTR, [])):
                    try:
                        fn()
                    except Exception as exc:
                        log(f"prompt-end CLIP/VAE cleanup: {exc}")

        execution.PromptExecutor.execute = wrapped
    if flush_fn not in fns:
        fns.append(flush_fn)


_MASTER_VAE_WORKERS = {}


def _register_vae_worker(conn):
    _MASTER_VAE_WORKERS[(conn.ip, conn.port)] = conn


def _flush_vae_workers():
    for (ip, port), conn in list(_MASTER_VAE_WORKERS.items()):
        try:
            log(f"Requesting VAE worker cleanup {ip}:{port}")
            resp, _ = conn.request(
                {"cmd": "cleanup", "proto": PROTOCOL_VERSION, "blob_size": 0},
                retries=1,
            )
            if isinstance(resp, dict) and resp.get("error"):
                log(f"VAE cleanup error from {ip}:{port}: {resp['error']}")
            else:
                log(f"VAE worker {ip}:{port} released GPU/RAM")
        except Exception as exc:
            log(f"VAE cleanup failed {ip}:{port}: {exc}")


def _restore_kwargs(kwargs):
    out = {}
    for k, v in (kwargs or {}).items():
        if isinstance(v, list):
            out[k] = tuple(v) if k in ("overlap",) else v
        else:
            out[k] = v
    return out


class _Connection:
    def __init__(self, ip, port, auth_token=""):
        self.ip = ip
        self.port = port
        self.auth_token = auth_token
        self._sock = None
        self._lock = threading.Lock()

    def _connect(self):
        last_err = None
        backoff = CONNECT_BACKOFF
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                sock = socket.create_connection((self.ip, self.port), timeout=SOCKET_TIMEOUT)
                set_socket_opts(sock)
                self._sock = sock
                log(f"Connected to worker {self.ip}:{self.port}")
                return
            except OSError as e:
                last_err = e
                log(f"Connect attempt {attempt}/{CONNECT_RETRIES} failed: {e}")
                if attempt < CONNECT_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
        raise ConnectionError(f"Could not connect to VAE worker {self.ip}:{self.port}: {last_err}")

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def request(self, header, blob=b"", response_timeout=None, retries=2):
        if self.auth_token:
            header = {**header, "auth": self.auth_token}
        with self._lock:
            for attempt in range(1, retries + 1):
                try:
                    if self._sock is None:
                        self._connect()
                    if response_timeout is not None:
                        self._sock.settimeout(response_timeout)
                    send_packet(self._sock, header, blob)
                    resp = recv_packet(self._sock)
                    self._sock.settimeout(IDLE_TIMEOUT)
                    return resp
                except (ConnectionError, OSError, struct.error, ValueError) as e:
                    log(f"Request failed ({e}); reconnecting (attempt {attempt}/{retries})")
                    self._close()
                    if attempt == retries:
                        raise


class RemoteVAE:
    """Drop-in stand-in for comfy.sd.VAE. Encode/decode run on the slave."""

    def __init__(self, ip, port, auth_token="", transport_mode="auto", connection=None, meta=None):
        self.ip = ip
        self.port = port
        self.transport_mode = transport_mode
        self.transport_dtype_name = resolve_transport_dtype(transport_mode, ip)
        self.transport_dtype = (
            ALLOWED_DTYPES[self.transport_dtype_name]
            if self.transport_dtype_name is not None
            else None
        )
        self._conn = connection or _Connection(ip, port, auth_token)
        self._meta = meta or self._fetch_meta()
        self.latent_dim = int(self._meta.get("latent_dim", 2))
        self.latent_channels = int(self._meta.get("latent_channels", 4))
        self.output_channels = int(self._meta.get("output_channels", 3))
        self.audio_sample_rate = int(self._meta.get("audio_sample_rate", 44100))
        self.audio_sample_rate_output = int(
            self._meta.get("audio_sample_rate_output", self.audio_sample_rate)
        )
        self.downscale_ratio = self._meta.get("spatial_encode", 8)
        self.upscale_ratio = self._meta.get("spatial_decode", 8)
        self.crop_input = bool(self._meta.get("crop_input", True))
        self.handles_tiling = bool(self._meta.get("handles_tiling", False))
        self.pad_channel_value = self._meta.get("pad_channel_value")
        self.extra_1d_channel = self._meta.get("extra_1d_channel")
        self.not_video = bool(self._meta.get("not_video", False))
        self.format_encoded = None
        self.first_stage_model = None
        self.patcher = None
        self.disable_offload = True
        self.size = 0

    def _fetch_meta(self):
        header = {"cmd": "meta", "proto": PROTOCOL_VERSION, "blob_size": 0}
        resp, _ = self._conn.request(header, retries=3)
        if resp.get("error"):
            raise RuntimeError(f"Remote VAE worker error: {resp['error']}")
        return resp.get("meta") or {}

    def throw_exception_if_invalid(self):
        if not self._meta:
            raise RuntimeError("Remote VAE has no worker metadata")

    def is_dynamic(self):
        return False

    def model_size(self):
        return 0

    def spacial_compression_decode(self):
        return int(self._meta.get("spatial_decode", 8))

    def spacial_compression_encode(self):
        return int(self._meta.get("spatial_encode", 8))

    def temporal_compression_decode(self):
        v = self._meta.get("temporal_decode")
        return int(v) if v is not None else None

    def _call(self, cmd, tensor, kwargs=None):
        self.throw_exception_if_invalid()
        meta, blob = pack_tensors({"x": tensor}, self.transport_dtype)
        header = {
            "cmd": cmd,
            "proto": PROTOCOL_VERSION,
            "tensors": meta,
            "kwargs": jsonable(kwargs or {}),
            "transport_dtype": self.transport_dtype_name,
            "blob_size": len(blob),
        }
        log(f"Sending {cmd} {tuple(tensor.shape)} ({len(blob)} bytes)")
        resp, out_blob = self._conn.request(
            header, blob, response_timeout=DECODE_TIMEOUT, retries=2
        )
        if resp.get("error"):
            raise RuntimeError(f"Remote VAE worker error: {resp['error']}")
        out = unpack_tensors(resp["tensors"], out_blob)["y"]
        device = comfy.model_management.intermediate_device()
        out = out.to(device)
        log(f"Received {cmd} {tuple(out.shape)} ({len(out_blob)} bytes)")
        return out

    def decode(self, samples_in, vae_options=None):
        return self._call("decode", samples_in, {"vae_options": vae_options or {}})

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        return self._call("decode_tiled", samples, {
            "tile_x": tile_x, "tile_y": tile_y, "overlap": overlap,
            "tile_t": tile_t, "overlap_t": overlap_t,
        })

    def encode(self, pixel_samples):
        return self._call("encode", pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        return self._call("encode_tiled", pixel_samples, {
            "tile_x": tile_x, "tile_y": tile_y, "overlap": overlap,
            "tile_t": tile_t, "overlap_t": overlap_t,
        })

    def get_sd(self):
        raise NotImplementedError("Remote VAE does not expose weights")


def _vae_meta(vae):
    temporal = None
    try:
        temporal = vae.temporal_compression_decode()
    except Exception:
        temporal = None
    return {
        "spatial_decode": int(vae.spacial_compression_decode()),
        "spatial_encode": int(vae.spacial_compression_encode()),
        "temporal_decode": temporal,
        "latent_dim": int(getattr(vae, "latent_dim", 2)),
        "latent_channels": int(getattr(vae, "latent_channels", 4)),
        "output_channels": int(getattr(vae, "output_channels", 3)),
        "audio_sample_rate": int(getattr(vae, "audio_sample_rate", 44100)),
        "audio_sample_rate_output": int(
            getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
        ),
        "crop_input": bool(getattr(vae, "crop_input", True)),
        "handles_tiling": bool(getattr(vae, "handles_tiling", False)),
        "pad_channel_value": getattr(vae, "pad_channel_value", None)
            if isinstance(getattr(vae, "pad_channel_value", None), (str, int, float, type(None)))
            else None,
        "extra_1d_channel": getattr(vae, "extra_1d_channel", None),
        "not_video": bool(getattr(vae, "not_video", False)),
    }


class _Worker:
    def __init__(self, vae, auth_token=""):
        self.vae = vae
        self.auth_token = auth_token
        self._infer_lock = threading.Lock()
        self.meta = _vae_meta(vae)

    def run(self, cmd, tensor, kwargs):
        kwargs = _restore_kwargs(kwargs)
        with self._infer_lock:
            with torch.inference_mode():
                if cmd == "decode":
                    return self.vae.decode(tensor, vae_options=kwargs.get("vae_options") or {})
                if cmd == "decode_tiled":
                    return self.vae.decode_tiled(
                        tensor,
                        tile_x=kwargs.get("tile_x"),
                        tile_y=kwargs.get("tile_y"),
                        overlap=kwargs.get("overlap"),
                        tile_t=kwargs.get("tile_t"),
                        overlap_t=kwargs.get("overlap_t"),
                    )
                if cmd == "encode":
                    return self.vae.encode(tensor)
                if cmd == "encode_tiled":
                    return self.vae.encode_tiled(
                        tensor,
                        tile_x=kwargs.get("tile_x"),
                        tile_y=kwargs.get("tile_y"),
                        overlap=kwargs.get("overlap"),
                        tile_t=kwargs.get("tile_t"),
                        overlap_t=kwargs.get("overlap_t"),
                    )
        raise ValueError(f"Unknown cmd {cmd}")

    def release_gpu(self):
        with self._infer_lock:
            _offload_vae_obj(self.vae)
        log("VAE worker: offloaded to CPU after request")

    def cleanup_after_job(self):
        with self._infer_lock:
            _offload_vae_obj(self.vae)
            _empty_device_cache()
        log("VAE worker: model off GPU, RAM trimmed")


class SendRemoteVAE:
    _servers = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "listen_port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
            },
            "optional": {
                "bind_host": ("STRING", {"default": "0.0.0.0"}),
                "auth_token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "start_worker"
    OUTPUT_NODE = True
    CATEGORY = "Remote VAE"

    def start_worker(self, vae, listen_port, bind_host="0.0.0.0", auth_token=""):
        token = auth_token or os.environ.get("REMOTE_VAE_TOKEN", "")
        existing = SendRemoteVAE._servers.pop(listen_port, None)
        if existing is not None:
            try:
                existing.close()
            except OSError:
                pass
        worker = _Worker(vae, token)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_host, listen_port))
        server.listen(8)
        SendRemoteVAE._servers[listen_port] = server
        if bind_host == "0.0.0.0" and not token:
            log("WARNING: worker bound to 0.0.0.0 with no auth token. "
                "Anyone on the LAN can use this VAE. Set auth_token or bind_host.")
        log(f"Worker listening on {bind_host}:{listen_port} "
            f"latent_dim={worker.meta['latent_dim']} ch={worker.meta['latent_channels']}")

        def handle(conn, addr):
            log(f"Client connected: {addr}")
            try:
                set_socket_opts(conn, timeout=IDLE_TIMEOUT)
                while True:
                    header = recv_header(conn)
                    if token and not hmac.compare_digest(header.get("auth", ""), token):
                        send_packet(conn, {"error": "unauthorized", "blob_size": 0})
                        log(f"Rejected unauthorized client {addr}")
                        break
                    if header.get("proto", 0) != PROTOCOL_VERSION:
                        send_packet(conn, {
                            "error": (f"protocol mismatch: worker speaks v{PROTOCOL_VERSION}, "
                                      f"client sent v{header.get('proto')}"),
                            "blob_size": 0,
                        })
                        break
                    blob_data = recv_blob(conn, header)
                    cmd = header.get("cmd")
                    if cmd == "meta":
                        send_packet(conn, {"meta": worker.meta, "blob_size": 0})
                        continue
                    if cmd == "cleanup":
                        try:
                            worker.cleanup_after_job()
                            send_packet(conn, {"ok": True, "blob_size": 0})
                        except Exception as e:
                            log(f"Cleanup failed: {e}")
                            send_packet(conn, {"error": str(e), "blob_size": 0})
                        continue
                    if cmd not in ("decode", "decode_tiled", "encode", "encode_tiled"):
                        send_packet(conn, {"error": "bad request", "blob_size": 0})
                        continue
                    try:
                        transport = header.get("transport_dtype")
                        transport_dtype = ALLOWED_DTYPES.get(transport) if transport else None
                        tensors = unpack_tensors(header.get("tensors", {}), blob_data)
                        out = worker.run(cmd, tensors["x"], header.get("kwargs") or {})
                        meta, blob = pack_tensors({"y": out}, transport_dtype)
                        send_packet(conn, {"tensors": meta, "blob_size": len(blob)}, blob)
                        log(f"Sent {cmd} result ({len(blob)} bytes)")
                        del tensors, out, blob, blob_data
                        worker.release_gpu()
                    except Exception as e:
                        log(f"{cmd} failed: {e}")
                        send_packet(conn, {"error": str(e), "blob_size": 0})
            except (ConnectionError, OSError) as e:
                log(f"Client {addr} disconnected: {e}")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        def accept_loop():
            while True:
                try:
                    conn, addr = server.accept()
                except OSError:
                    log("Server socket closed; stopping accept loop")
                    break
                threading.Thread(target=handle, args=(conn, addr), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        return {"ui": {"text": [f"Remote VAE worker on {bind_host}:{listen_port}"]}}


class LoadRemoteVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "worker_ip": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
            },
            "optional": {
                "auth_token": ("STRING", {"default": ""}),
                "transport_precision": (["auto", "fp16", "fp32"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_remote"
    CATEGORY = "Remote VAE"

    def load_remote(self, worker_ip, port, auth_token="", transport_precision="auto"):
        token = auth_token or os.environ.get("REMOTE_VAE_TOKEN", "")
        vae = RemoteVAE(worker_ip, port, token, transport_precision)
        _register_vae_worker(vae._conn)
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "SendRemoteVAE": SendRemoteVAE,
    "LoadRemoteVAE": LoadRemoteVAE,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SendRemoteVAE": "Send Remote VAE",
    "LoadRemoteVAE": "Load Remote VAE",
}

try:
    _install_master_cleanup_hook(_flush_vae_workers)
except Exception as exc:
    log(f"Could not install prompt-end VAE cleanup hook: {exc}")
