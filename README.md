# ComfyUI Remote VAE

Run **VAE encode and decode on a slave machine**. The master keeps the diffusion model / sampler in VRAM and only sends latents (and pixels for encode) over the LAN.

This is the VAE counterpart of **[ComfyUI-RemoteCLIPLoader](https://github.com/nyueki/ComfyUI-RemoteCLIPLoader)** by [nyueki](https://github.com/nyueki). Use that plugin to offload CLIP / text encoders; use this one to offload video and audio VAEs. The wire protocol is the same idea (JSON header + tensor blob, optional shared `auth_token`) but the ports and nodes are separate, so both can run at once.

Typical split for MiniMax H3:

- **Master** — UNET / DiT sampling (and optionally CLIP via Remote CLIP).
- **Slave** — `minimax_h3_video_vae` on port **8182** and `minimax_h3_audio_vae` on port **8183**.

After sampling, a local `VAEDecode` would otherwise load a multi-gigabyte video VAE onto the same GPUs that just ran the DiT. Remote VAE keeps those weights on the slave.

## Roles

| Role | Machine | Nodes |
|---|---|---|
| **Slave (sender)** | Holds the `.safetensors` VAE file(s) and does the math | `VAELoader` → **Send Remote VAE** |
| **Master (loader)** | Runs the full graph | **Load Remote VAE** anywhere a `VAE` socket is expected |

Install the **same plugin version** on both machines.

## Installation

On **each** ComfyUI (`custom_nodes`):

```bash
git clone https://github.com/wealllovegithub/Comfyui_remote_vae.git
```

Restart ComfyUI.

Copy the VAE weights you will serve onto the **slave** only, under `ComfyUI/models/vae/`. Example for MiniMax H3:

- `minimax_h3_video_vae_fp16.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`

The master does **not** need those files if every VAE socket is a Load Remote VAE node.

## Usage

### Slave

1. Add **VAE Loader** and pick the VAE.
2. Connect it to **Send Remote VAE**.
3. Set `listen_port` (default **8182**). Use a second Send node on **8183** for a second VAE (audio).
4. Optional: `auth_token` (shared secret) and `bind_host` (default `0.0.0.0`).
5. **Queue the prompt once.** The node is an output node; it keeps listening until ComfyUI restarts. Re-queue after a restart.

Example graph (also in `example_workflows/slave_minimax_h3_vaes.json`):

```
VAELoader (video vae)  →  Send Remote VAE   port 8182
VAELoader (audio vae)  →  Send Remote VAE   port 8183
```

Open the ports on the slave firewall, **LAN only**:

```bash
# Linux (UFW), replace the subnet with yours
sudo ufw allow from 192.168.1.0/24 to any port 8182 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8183 proto tcp
```

Windows: allow TCP 8182/8183 from your LAN in Windows Defender Firewall.

### Master

1. Add **Load Remote VAE**.
2. Set `worker_ip` to the slave’s LAN IP and `port` to match the sender.
3. Connect the output to **VAE Decode**, **VAE Decode Audio**, **MiniMax H3 Image to Video**, or any other `VAE` input.

| What | Load Remote VAE |
|---|---|
| Video decode / keyframe encode | slave IP, port **8182** |
| Audio decode | slave IP, port **8183** |

Leave the sampler on the master. CLIP can stay local or use [Remote CLIP](https://github.com/nyueki/ComfyUI-RemoteCLIPLoader) on its own port (default 8181).

Optional:

- `auth_token` — must match the sender. Can also come from the `REMOTE_VAE_TOKEN` environment variable on either side.
- `transport_precision` — `auto` (default) sends fp16 on the LAN and full precision on localhost; `fp16` always half; `fp32` never downcasts. Use `fp32` for audio if you hear quantization.

## Together with Remote CLIP

| Plugin | Default port | Offloads |
|---|---|---|
| [ComfyUI-RemoteCLIPLoader](https://github.com/nyueki/ComfyUI-RemoteCLIPLoader) | 8181 | CLIP / text encoder (and generation-capable encoders) |
| This repo | 8182 (video), 8183 (audio) | VAE encode / decode |

Do not put Send Remote CLIP and Send Remote VAE on the same port.

## Notes

- Traffic is **not encrypted**. Use a LAN or a VPN. Do not expose these ports to the internet.
- Protocol v1 must match on both sides.
- The worker keeps each client socket for **4 hours** of idle time so the master can finish a long sample before `VAEDecode`. A short timeout will drop the link mid-job (`Socket closed while reading`).
- One encode/decode at a time per worker.
- Video decode can be 0.5–2 GB per clip. Gigabit Ethernet is fine; Wi-Fi is painful.
- MiniMax H3 already tiles inside `decode()`. On the master, prefer **VAE Decode** over **VAE Decode (Tiled)**.
- Re-queue **Send Remote VAE** after every ComfyUI restart on the slave.

Node category: **Remote VAE**

## License

MIT. Protocol layout inspired by [ComfyUI-RemoteCLIPLoader](https://github.com/nyueki/ComfyUI-RemoteCLIPLoader).
