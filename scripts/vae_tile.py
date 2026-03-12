#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/05
# Modified: Forge Classic compatibility
#   - No ldm dependency
#   - Hooks VAE.decode / VAE.encode (backend/vae.py) instead of encoder/decoder.forward
#   - Patch sd_vae._load_vae_dict to survive VAE reloads

import math
from time import time
from traceback import print_exc
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
import gradio as gr

import modules.devices as devices
from modules.scripts import Script, AlwaysVisible
from modules.shared import state
from modules.ui import gr_show
from modules.processing import StableDiffusionProcessing

from typing import Tuple, List
from torch import Tensor


# ── Default tile sizes based on VRAM ──────────────────────────────────────────

def get_default_encoder_tile_size() -> int:
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if   mem > 16000: return 3072
        elif mem > 12000: return 2048
        elif mem > 10000: return 1536
        elif mem >  8000: return 1536
        elif mem >  6000: return 1024
        elif mem >  4000: return 768
        else:             return 512
    return 512

def get_default_decoder_tile_size() -> int:
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if   mem > 30000: return 256
        elif mem > 16000: return 192
        elif mem > 12000: return 128
        elif mem > 10000: return 96
        elif mem >  8000: return 96
        elif mem >  6000: return 80
        elif mem >  4000: return 64
        else:             return 64
    return 64


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_OPEN              =  True
DEFAULT_ENABLED           = True
DEFAULT_SMART_IGNORE      = False
DEFAULT_ENCODER_PAD_SIZE  = 16
DEFAULT_DECODER_PAD_SIZE  = 4
DEFAULT_ENCODER_TILE_SIZE = 512
DEFAULT_DECODER_TILE_SIZE = 16

BBox = Tuple[int, int, int, int]


# ── Tiled VAE core ─────────────────────────────────────────────────────────────

def _calc_tile_size(size: int, tile_size: int) -> int:
    """Return real tile size for one dimension (split evenly)."""
    n = math.ceil(size / tile_size)
    return math.ceil(size / n)

def _make_bboxes(
    H: int, W: int,
    tile_H: int, tile_W: int,
    P: int,
    scaler: float,
) -> Tuple[List[BBox], List[BBox]]:
    """Build input/output bounding boxes for all tiles."""
    inputs, outputs = [], []
    x = 0
    while x < H:
        y = 0
        while y < W:
            x2 = min(x + tile_H, H)
            y2 = min(y + tile_W, W)
            inputs.append((x, x2 + 2*P, y, y2 + 2*P))
            outputs.append((
                round(x  * scaler), round(x2 * scaler),
                round(y  * scaler), round(y2 * scaler),
            ))
            y += tile_W
        x += tile_H
    return inputs, outputs


@torch.inference_mode()
def tiled_decode(vae_obj, samples: Tensor, tile_size: int, pad_size: int) -> Tensor:
    """
    Decode latent tensor tile by tile.
    Returns [B, 3, H*upscale, W*upscale] float32 on cpu, values in decoder raw output space.
    """
    B, C, H, W = samples.shape
    P = pad_size
    upscale = vae_obj.upscale_ratio   # 8 for SD/SDXL

    tile_H = _calc_tile_size(H, tile_size)
    tile_W = _calc_tile_size(W, tile_size)
    n_H = math.ceil(H / tile_H)
    n_W = math.ceil(W / tile_W)
    n_tiles = n_H * n_W

    print(f'[VAE Tile] decode: input {list(samples.shape)}, '
          f'tile {tile_H}x{tile_W}, grid {n_H}x{n_W}={n_tiles}, pad {P}')

    padded = F.pad(samples, (P, P, P, P), mode='reflect') if P != 0 else samples

    bbox_in, bbox_out = _make_bboxes(H, W, tile_H, tile_W, P, upscale)

    out_H = round(H * upscale)
    out_W = round(W * upscale)
    result     = torch.zeros([B, 3, out_H, out_W], dtype=torch.float32, device='cpu')
    result_cnt = torch.zeros([B, 1, out_H, out_W], dtype=torch.float32, device='cpu')

    pbar = tqdm(total=n_tiles, desc='VAE tile decoding')
    for (xs, xe, ys, ye), (oxs, oxe, oys, oye) in zip(bbox_in, bbox_out):
        if state.interrupted:
            break

        tile = padded[:, :, xs:xe, ys:ye].to(vae_obj.vae_dtype).to(vae_obj.device)
        decoded = vae_obj.first_stage_model.decode(tile).float().cpu()
        # decoded: [B, 3, tile_out_H, tile_out_W]

        # crop padding from output side
        if P != 0:
            op = round(P * upscale)
            th, tw = decoded.shape[2], decoded.shape[3]
            decoded = decoded[:, :, op:max(op+1, th-op), op:max(op+1, tw-op)]

        # clamp to actual output region (edge tiles may be smaller)
        ah = oxe - oxs
        aw = oye - oys
        decoded = decoded[:, :, :ah, :aw]

        result    [:, :, oxs:oxe, oys:oye] += decoded
        result_cnt[:, :, oxs:oxe, oys:oye] += 1.0
        pbar.update()

    pbar.close()
    result = result / result_cnt.clamp(min=1e-6)
    return result   # raw decoder output, caller does process_output


@torch.inference_mode()
def tiled_encode(vae_obj, pixel_samples: Tensor, tile_size: int, pad_size: int) -> Tensor:
    """
    Encode pixel tensor tile by tile.
    pixel_samples: [B, 3, H, W] float in [0,1]
    Returns latent [B, C, H/downscale, W/downscale] on output_device.
    """
    B, C, H, W = pixel_samples.shape
    P = pad_size
    downscale = vae_obj.downscale_ratio   # 8 for SD/SDXL

    tile_H = _calc_tile_size(H, tile_size)
    tile_W = _calc_tile_size(W, tile_size)
    n_H = math.ceil(H / tile_H)
    n_W = math.ceil(W / tile_W)
    n_tiles = n_H * n_W

    print(f'[VAE Tile] encode: input {list(pixel_samples.shape)}, '
          f'tile {tile_H}x{tile_W}, grid {n_H}x{n_W}={n_tiles}, pad {P}')

    padded = F.pad(pixel_samples, (P, P, P, P), mode='reflect') if P != 0 else pixel_samples

    bbox_in, bbox_out = _make_bboxes(H, W, tile_H, tile_W, P, 1.0 / downscale)

    out_H = round(H / downscale)
    out_W = round(W / downscale)
    latent_ch  = vae_obj.latent_channels
    result     = torch.zeros([B, latent_ch, out_H, out_W], dtype=torch.float32, device='cpu')
    result_cnt = torch.zeros([B, 1,         out_H, out_W], dtype=torch.float32, device='cpu')

    pbar = tqdm(total=n_tiles, desc='VAE tile encoding')
    for (xs, xe, ys, ye), (oxs, oxe, oys, oye) in zip(bbox_in, bbox_out):
        if state.interrupted:
            break

        tile = padded[:, :, xs:xe, ys:ye]
        tile = vae_obj.process_input(tile).to(vae_obj.vae_dtype).to(vae_obj.device)
        encoded = vae_obj.first_stage_model.encode(tile).float().cpu()

        if P != 0:
            op = round(P / downscale)
            th, tw = encoded.shape[2], encoded.shape[3]
            encoded = encoded[:, :, op:max(op+1, th-op), op:max(op+1, tw-op)]

        ah = oxe - oxs
        aw = oye - oys
        encoded = encoded[:, :, :ah, :aw]

        result    [:, :, oxs:oxe, oys:oye] += encoded
        result_cnt[:, :, oxs:oxe, oys:oye] += 1.0
        pbar.update()

    pbar.close()
    result = result / result_cnt.clamp(min=1e-6)
    return result.to(vae_obj.output_device)


# ── Perf wrapper ───────────────────────────────────────────────────────────────

def perfcount(fn):
    def wrapper(*args, **kwargs):
        device = devices.device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        devices.torch_gc()
        gc.collect()
        ts = time()
        try:
            return fn(*args, **kwargs)
        except:
            raise
        finally:
            te = time()
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated(device) / 2**20
                torch.cuda.reset_peak_memory_stats(device)
                print(f'[VAE Tile] Done in {te-ts:.3f}s, max VRAM {vram:.1f} MB')
            else:
                print(f'[VAE Tile] Done in {te-ts:.3f}s')
            devices.torch_gc()
            gc.collect()
    return wrapper


# ── Patched VAE.decode / VAE.encode ───────────────────────────────────────────

@perfcount
def patched_decode(vae_obj, samples_in: Tensor,
                   tile_size: int, pad_size: int, smart_ignore: bool) -> Tensor:
    B, C, H, W = samples_in.shape

    if smart_ignore and max(H, W) < tile_size:
        print(f'[VAE Tile] decode: {H}x{W} < tile_size {tile_size}, skipping tiling')
        return vae_obj._orig_decode(samples_in)

    from backend import memory_management
    memory_management.load_models_gpu([vae_obj.patcher])

    raw = tiled_decode(vae_obj, samples_in, tile_size, pad_size)
    # raw is in decoder native output space (typically [-1,1] before process_output)
    # process_output does clamp((x+1)/2, 0, 1), but our tiles already came through
    # first_stage_model.decode which may or may not apply that transform.
    # We apply process_output here to be safe, then movedim for Forge's expected format.
    out = vae_obj.process_output(raw)               # [B, 3, H, W] -> [0,1]
    return out.movedim(1, -1).to(vae_obj.output_device)   # [B, H, W, 3]


@perfcount
def patched_encode(vae_obj, pixel_samples: Tensor,
                   tile_size: int, pad_size: int, smart_ignore: bool) -> Tensor:
    # Forge passes [B, H, W, 3]; convert to [B, 3, H, W]
    _s = pixel_samples.movedim(-1, 1)
    B, C, H, W = _s.shape

    if smart_ignore and max(H, W) < tile_size:
        print(f'[VAE Tile] encode: {H}x{W} < tile_size {tile_size}, skipping tiling')
        return vae_obj._orig_encode(pixel_samples)

    from backend import memory_management
    memory_management.load_models_gpu([vae_obj.patcher])

    return tiled_encode(vae_obj, _s, tile_size, pad_size)


# ── Script ────────────────────────────────────────────────────────────────────

_current_script_instance = None


class Script(Script):

    def title(self):
        return "Yet Another VAE Tiling"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Yet Another VAE Tiling', open=DEFAULT_OPEN):
            with gr.Row(variant='compact'):
                enabled = gr.Checkbox(label='Enabled', value=lambda: DEFAULT_ENABLED)
                reset   = gr.Button(value='↻', variant='tool')

            with gr.Row(variant='compact'):
                encoder_tile_size = gr.Slider(label='Encoder tile size',
                    minimum=512, maximum=4096, step=32, value=lambda: DEFAULT_ENCODER_TILE_SIZE)
                decoder_tile_size = gr.Slider(label='Decoder tile size',
                    minimum=8,   maximum=256,  step=1,  value=lambda: DEFAULT_DECODER_TILE_SIZE)

            with gr.Row(variant='compact'):
                encoder_pad_size = gr.Slider(label='Encoder pad size',
                    minimum=0, maximum=64, step=8, value=lambda: DEFAULT_ENCODER_PAD_SIZE)
                decoder_pad_size = gr.Slider(label='Decoder pad size',
                    minimum=0, maximum=8,  step=1, value=lambda: DEFAULT_DECODER_PAD_SIZE)

            reset.click(
                fn=lambda: [DEFAULT_ENCODER_TILE_SIZE, DEFAULT_ENCODER_PAD_SIZE,
                            DEFAULT_DECODER_TILE_SIZE, DEFAULT_DECODER_PAD_SIZE],
                outputs=[encoder_tile_size, encoder_pad_size,
                         decoder_tile_size, decoder_pad_size],
            )

            with gr.Row(variant='compact'):
                ext_smart_ignore = gr.Checkbox(
                    label='Do not process small images', value=lambda: DEFAULT_SMART_IGNORE)

        return [enabled,
                encoder_tile_size, encoder_pad_size,
                decoder_tile_size, decoder_pad_size,
                ext_smart_ignore]

    def process(self, p: StableDiffusionProcessing,
                enabled: bool,
                encoder_tile_size: int, encoder_pad_size: int,
                decoder_tile_size: int, decoder_pad_size: int,
                ext_smart_ignore: bool):

        global _current_script_instance
        _current_script_instance = self

        self._tile_enabled  = enabled
        self._tile_enc_tile = encoder_tile_size
        self._tile_enc_pad  = encoder_pad_size
        self._tile_dec_tile = decoder_tile_size
        self._tile_dec_pad  = decoder_pad_size
        self._tile_smart    = ext_smart_ignore

        # Try to hook now (works if VAE already in memory from previous run)
        self._apply_hijack(p)

    def _get_vae(self, p):
        """
        Locate the backend.vae.VAE instance.
        Forge Classic exposes it via p.sd_model.forge_objects.vae.
        """
        candidates = [
            lambda: p.sd_model.forge_objects.vae,
            lambda: p.sd_model.vae,
        ]
        for getter in candidates:
            try:
                obj = getter()
                if obj is not None and hasattr(obj, 'decode') and hasattr(obj, 'encode') \
                        and hasattr(obj, 'first_stage_model'):
                    return obj
            except AttributeError:
                continue
        return None

    def _apply_hijack(self, p):
        if not hasattr(self, '_tile_enabled'):
            return

        enabled           = self._tile_enabled
        encoder_tile_size = self._tile_enc_tile
        encoder_pad_size  = self._tile_enc_pad
        decoder_tile_size = self._tile_dec_tile
        decoder_pad_size  = self._tile_dec_pad
        smart_ignore      = self._tile_smart

        vae = self._get_vae(p)
        if vae is None:
            print('[VAE Tile] Cannot locate VAE object, skipping.')
            return

        # ── restore originals when disabled ───────────────────────────────────
        if not enabled:
            if hasattr(vae, '_orig_decode'):
                vae.decode = vae._orig_decode
                del vae._orig_decode
            if hasattr(vae, '_orig_encode'):
                vae.encode = vae._orig_encode
                del vae._orig_encode
            return

        # ── save originals once per vae instance ──────────────────────────────
        if not hasattr(vae, '_orig_decode'):
            vae._orig_decode = vae.decode
        if not hasattr(vae, '_orig_encode'):
            vae._orig_encode = vae.encode

        # ── apply patch ───────────────────────────────────────────────────────
        vae.decode = lambda s: patched_decode(
            vae, s, decoder_tile_size, decoder_pad_size, smart_ignore)
        vae.encode = lambda s: patched_encode(
            vae, s, encoder_tile_size, encoder_pad_size, smart_ignore)

        print(f'[VAE Tile] hijack applied on {vae.__class__.__name__}: '
              f'enc_tile={encoder_tile_size} dec_tile={decoder_tile_size}')

    def postprocess(self, p, processed, *args):
        # Re-apply after first inference:
        # Forge loads VAE *after* process(), so the first hook may miss.
        # postprocess() runs after VAE is guaranteed in memory.
        self._apply_hijack(p)


# ── Patch sd_vae._load_vae_dict to survive VAE weight reloads ─────────────────

def _re_apply_after_vae_load(model):
    if _current_script_instance is None:
        return
    try:
        class _FakeP:
            sd_model = model
        _current_script_instance._apply_hijack(_FakeP())
    except Exception as e:
        print(f'[VAE Tile] re-apply after VAE load failed: {e}')


def _patch_sd_vae():
    try:
        from modules import sd_vae
        _orig = sd_vae._load_vae_dict

        def _patched(model, vae_sd):
            _orig(model, vae_sd)
            print('[VAE Tile] VAE weights reloaded, re-applying hijack...')
            _re_apply_after_vae_load(model)

        sd_vae._load_vae_dict = _patched
        print('[VAE Tile] sd_vae._load_vae_dict patched OK')
    except Exception as e:
        print(f'[VAE Tile] sd_vae patch failed (non-fatal): {e}')


_patch_sd_vae()
