"""
esrgan.py - Real-ESRGAN x4 upscaling step for jersey number crops.

Sits in the pipeline between pose-guided crop generation (step 6) and
PARSeq scene-text recognition (step 7).  Small jersey-number crops are
notoriously low-resolution; upscaling them 4× before feeding to PARSeq
improves digit legibility and STR accuracy.

Usage (standalone):
    python esrgan.py --input_dir  out/SoccerNetResults/test/crops/imgs \
                     --output_dir out/SoccerNetResults/test/crops_sr/imgs \
                     --model_path models/RealESRGAN_x4plus.pth \
                     --scale 4 \
                     --tile 0 \
                     --batch_size 8

The module also exposes `upscale_directory()` so it can be called directly
from main.py without spawning a subprocess.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Lazy import of basicsr / realesrgan so the rest of the pipeline still works
# even if the package has not been installed yet.
# ---------------------------------------------------------------------------
def _import_realesrgan():
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        return RRDBNet, RealESRGANer, SRVGGNetCompact
    except ImportError as exc:
        raise ImportError(
            "Real-ESRGAN is not installed.  Run  setup.py  or:\n"
            "  pip install realesrgan basicsr\n"
        ) from exc


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = {
    "RealESRGAN_x4plus": {
        "scale": 4,
        "num_block": 23,
        "num_feat": 64,
        "arch": "RRDBNet",
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRGAN_x2plus": {
        "scale": 2,
        "num_block": 23,
        "num_feat": 64,
        "arch": "RRDBNet",
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
    "realesr-general-x4v3": {
        "scale": 4,
        "num_block": None,
        "num_feat": 64,
        "arch": "SRVGGNetCompact",
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    },
}


def build_upsampler(model_path: str, scale: int = 4,
                    tile: int = 0, tile_pad: int = 10,
                    pre_pad: int = 0, half: bool = False,
                    device: str | None = None):
    """
    Instantiate a RealESRGANer upsampler from a checkpoint path.

    The model architecture is inferred from the filename; fall back to the
    standard RealESRGAN_x4plus (RRDBNet-23) if the name is unrecognised.
    """
    RRDBNet, RealESRGANer, SRVGGNetCompact = _import_realesrgan()

    basename = os.path.basename(model_path)
    # Strip extension for lookup
    key = os.path.splitext(basename)[0]
    spec = SUPPORTED_MODELS.get(key, SUPPORTED_MODELS["RealESRGAN_x4plus"])

    if spec["arch"] == "SRVGGNetCompact":
        net = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=spec["num_feat"],
            num_conv=32, upscale=spec["scale"],
            act_type="prelu",
        )
    else:
        net = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=spec["num_feat"],
            num_block=spec["num_block"],
            num_grow_ch=32,
            scale=spec["scale"],
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=net,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        device=device,
    )
    return upsampler


# ---------------------------------------------------------------------------
# Core upscaling routine
# ---------------------------------------------------------------------------
def upscale_directory(
    input_dir: str,
    output_dir: str,
    model_path: str,
    scale: int = 4,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    half: bool = False,
    batch_size: int = 1,
    ext: str = "auto",
    overwrite: bool = False,
    intermediate_dir: str | None = None,
) -> int:
    """
    Upscale every image in *input_dir* and save results to *output_dir*.

    Parameters
    ----------
    input_dir        : directory containing the pose-guided crops (imgs/ folder)
    output_dir       : destination for super-resolved images (same filenames)
    model_path       : path to the Real-ESRGAN .pth checkpoint
    scale            : upscaling factor (4 for RealESRGAN_x4plus)
    tile             : tile size for patch-based inference (0 = whole image)
    tile_pad         : padding between tiles
    pre_pad          : padding added before processing
    half             : use FP16 inference (faster on modern GPUs)
    batch_size       : images processed per GPU batch (kept for API compat;
                       RealESRGANer processes one image at a time internally)
    ext              : output extension – "auto" keeps the original extension
    overwrite        : skip images that already exist in output_dir
    intermediate_dir : if set, the raw numpy array returned by enhance() is
                       saved here as <stem>.npy immediately after each call,
                       before cv2.imwrite() converts it to an image file.
                       Useful for debugging or downstream processing that needs
                       the uncompressed float/uint8 array.  Directory is
                       created automatically if it does not exist.

    Returns
    -------
    Number of images successfully upscaled.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if intermediate_dir is not None:
        Path(intermediate_dir).mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in supported_exts
    ])

    if not image_files:
        print(f"[ESRGAN] No images found in {input_dir}")
        return 0

    print(f"[ESRGAN] Loading model from {model_path} ...")
    upsampler = build_upsampler(
        model_path=model_path, scale=scale,
        tile=tile, tile_pad=tile_pad,
        pre_pad=pre_pad, half=half,
    )

    def _save_intermediate(array: np.ndarray, stem: str) -> None:
        """Save the raw numpy array from enhance() to intermediate_dir."""
        if intermediate_dir is None:
            return
        npy_path = os.path.join(intermediate_dir, stem + ".npy")
        np.save(npy_path, array)

    success_count = 0
    for filename in tqdm(image_files, desc="Real-ESRGAN upscaling"):
        # Determine output filename
        stem, orig_ext = os.path.splitext(filename)
        out_ext = orig_ext if ext == "auto" else ("." + ext.lstrip("."))
        out_filename = stem + out_ext
        out_path = os.path.join(output_dir, out_filename)

        if not overwrite and os.path.exists(out_path):
            success_count += 1
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ESRGAN] Warning: could not read {img_path}, skipping.")
            continue

        try:
            # enhance() expects BGR (OpenCV format) and returns BGR
            output, _ = upsampler.enhance(img, outscale=scale)
            _save_intermediate(output, stem)  # save raw array before imwrite
            cv2.imwrite(out_path, output)
            success_count += 1
        except RuntimeError as e:
            # Fallback: CUDA OOM → retry on CPU with tiling
            if "out of memory" in str(e).lower():
                print(f"[ESRGAN] CUDA OOM on {filename}; retrying with tile=256 on CPU.")
                torch.cuda.empty_cache()
                cpu_upsampler = build_upsampler(
                    model_path=model_path, scale=scale,
                    tile=256, tile_pad=tile_pad,
                    pre_pad=pre_pad, half=False,
                    device="cpu",
                )
                try:
                    output, _ = cpu_upsampler.enhance(img, outscale=scale)
                    _save_intermediate(output, stem)  # save raw array before imwrite
                    cv2.imwrite(out_path, output)
                    success_count += 1
                except Exception as e2:
                    print(f"[ESRGAN] CPU fallback also failed for {filename}: {e2}")
            else:
                print(f"[ESRGAN] Error processing {filename}: {e}")

    print(f"[ESRGAN] Upscaled {success_count}/{len(image_files)} images → {output_dir}")
    return success_count


# ---------------------------------------------------------------------------
# Model weight downloader
# ---------------------------------------------------------------------------
def download_model(model_name: str = "RealESRGAN_x4plus",
                   save_dir: str = "models") -> str:
    """Download the Real-ESRGAN weights if they are not already present."""
    import urllib.request

    spec = SUPPORTED_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from: {list(SUPPORTED_MODELS.keys())}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, spec["filename"])

    if os.path.isfile(save_path):
        print(f"[ESRGAN] Model already present at {save_path}")
        return save_path

    print(f"[ESRGAN] Downloading {model_name} weights from {spec['url']} ...")
    urllib.request.urlretrieve(spec["url"], save_path)
    print(f"[ESRGAN] Saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-ESRGAN x4 upscaler for jersey-number crops"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing input crops (imgs/ folder)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write upscaled images")
    parser.add_argument("--model_path", default="models/RealESRGAN_x4plus.pth",
                        help="Path to Real-ESRGAN .pth checkpoint")
    parser.add_argument("--scale", type=int, default=4,
                        help="Upscaling factor (default: 4)")
    parser.add_argument("--tile", type=int, default=0,
                        help="Tile size for patch inference; 0 = whole image")
    parser.add_argument("--tile_pad", type=int, default=10,
                        help="Overlap padding between tiles (default: 10)")
    parser.add_argument("--pre_pad", type=int, default=0,
                        help="Pre-padding before processing (default: 0)")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 inference (faster on modern GPUs)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size hint (default: 1)")
    parser.add_argument("--ext", default="auto",
                        help="Output extension, e.g. png; 'auto' keeps original")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process images that already exist in output_dir")
    parser.add_argument("--intermediate_dir", default=None,
                        help="If set, save the raw numpy array from enhance() here as "
                             "<stem>.npy before writing the final image (useful for "
                             "debugging or lossless downstream processing)")
    parser.add_argument("--download_only", action="store_true",
                        help="Just download the model weights and exit")

    args = parser.parse_args()

    if args.download_only:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        save_dir = os.path.dirname(args.model_path) or "models"
        download_model(model_name, save_dir)
    else:
        upscale_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            scale=args.scale,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=args.half,
            batch_size=args.batch_size,
            ext=args.ext,
            overwrite=args.overwrite,
            intermediate_dir=args.intermediate_dir,
        )