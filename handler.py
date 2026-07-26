#!/usr/bin/env python3
"""
Runpod Serverless Real-ESRGAN image upscaler.

Expected Network Volume mount:
    /runpod-volume

Input object example:
{
  "input": {
    "source": "upscale/input/photo.jpg",
    "scale": 4,
    "format": "png",
    "tile": 0
  }
}

The worker reads:
    /runpod-volume/<source>

and writes:
    /runpod-volume/upscale/output/<job-id>/<filename>

The response returns the relative object key so the caller can fetch it
through the Runpod Network Volume S3-compatible API.
"""

import os
import sys
import types
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# torchvision compatibility shim
# ---------------------------------------------------------------------------
# basicsr/realesrgan imports torchvision.transforms.functional_tensor, which
# no longer exists in newer torchvision releases.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    import torchvision.transforms.functional as _F

    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim

import cv2
import numpy as np
from PIL import Image
import runpod
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume"))
UPSCALE_ROOT = VOLUME_ROOT / "upscale"
MODELS_DIR = UPSCALE_ROOT / "models"
OUTPUT_DIR = UPSCALE_ROOT / "output"

MODEL_NAME = "RealESRGAN_x4plus.pth"
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
NATIVE_SCALE = 4
SUPPORTED_INPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_FORMATS = {"png", "jpg"}

# Create persistent directories at worker startup.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_weights() -> Path:
    weights_path = MODELS_DIR / MODEL_NAME

    if not weights_path.exists():
        print(f"Downloading {MODEL_NAME} to {weights_path}", flush=True)
        urlretrieve(MODEL_URL, weights_path)

    return weights_path


WEIGHTS_PATH = ensure_weights()

# Cache upsamplers by tile size so multiple jobs on a warm worker do not reload
# the model unnecessarily.
_UPSAMPLERS: dict[int, RealESRGANer] = {}


def get_upsampler(tile: int) -> RealESRGANer:
    if tile in _UPSAMPLERS:
        return _UPSAMPLERS[tile]

    print(
        f"Loading Real-ESRGAN on {'CUDA' if torch.cuda.is_available() else 'CPU'} "
        f"(tile={tile})",
        flush=True,
    )

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=NATIVE_SCALE,
    )

    upsampler = RealESRGANer(
        scale=NATIVE_SCALE,
        model_path=str(WEIGHTS_PATH),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )

    _UPSAMPLERS[tile] = upsampler
    return upsampler


def resolve_source(source_key: str) -> Path:
    """
    Resolve a relative Network Volume key safely beneath /runpod-volume.
    """
    source_key = source_key.lstrip("/")
    source_path = (VOLUME_ROOT / source_key).resolve()
    volume_root = VOLUME_ROOT.resolve()

    try:
        source_path.relative_to(volume_root)
    except ValueError as exc:
        raise ValueError("source must stay inside the Network Volume") from exc

    if not source_path.is_file():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    if source_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported source extension: {source_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    return source_path


def upscale_image(
    source_path: Path,
    destination_path: Path,
    outscale: float,
    output_format: str,
    tile: int,
) -> tuple[int, int, int, int]:
    with Image.open(source_path) as image:
        rgb = np.array(image.convert("RGB"))

    source_height, source_width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    upsampler = get_upsampler(tile)
    output_bgr, _ = upsampler.enhance(bgr, outscale=outscale)

    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_rgb)

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "png":
        output_image.save(destination_path)
    else:
        output_image.save(destination_path, quality=95)

    return (
        source_width,
        source_height,
        output_image.width,
        output_image.height,
    )


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}

    source_key = job_input.get("source")
    if not source_key:
        raise ValueError(
            'Missing required input field "source", for example '
            '"upscale/input/photo.jpg"'
        )

    scale = float(job_input.get("scale", 4.0))
    if scale <= 0:
        raise ValueError("scale must be greater than 0")

    output_format = str(job_input.get("format", "png")).lower()
    if output_format == "jpeg":
        output_format = "jpg"
    if output_format not in SUPPORTED_FORMATS:
        raise ValueError('format must be "png" or "jpg"')

    tile = int(job_input.get("tile", 0))
    if tile < 0:
        raise ValueError("tile must be 0 or a positive integer")

    source_path = resolve_source(str(source_key))

    job_id = str(job.get("id") or "manual")
    suffix = ".png" if output_format == "png" else ".jpg"

    destination_dir = OUTPUT_DIR / job_id
    destination_path = destination_dir / f"{source_path.stem}{suffix}"

    print(
        f"Upscaling {source_path} -> {destination_path} "
        f"(scale={scale}, format={output_format}, tile={tile})",
        flush=True,
    )

    (
        source_width,
        source_height,
        output_width,
        output_height,
    ) = upscale_image(
        source_path=source_path,
        destination_path=destination_path,
        outscale=scale,
        output_format=output_format,
        tile=tile,
    )

    # Return a Network Volume/S3 object key, not an internal container path.
    output_key = destination_path.relative_to(VOLUME_ROOT).as_posix()

    return {
        "output_key": output_key,
        "source": str(source_key),
        "source_width": source_width,
        "source_height": source_height,
        "width": output_width,
        "height": output_height,
        "scale": scale,
        "format": output_format,
    }


if __name__ == "__main__":
    print(
        f"Starting Runpod Real-ESRGAN worker; volume={VOLUME_ROOT}; "
        f"cuda={torch.cuda.is_available()}",
        flush=True,
    )

    if os.environ.get("LOCAL_TEST") == "1":
        test_job = {
            "id": "local-test",
            "input": {
                "source": "upscale/input/graccius-brothers-3.webp",
                "scale": 4,
                "format": "png",
                "tile": 0,
            },
        }

        result = handler(test_job)

        print("Local test result:")
        print(result)
    else:
        runpod.serverless.start({"handler": handler})
