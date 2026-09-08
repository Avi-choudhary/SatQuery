from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

CHANGEFORMER_DIR = (
    BACKEND_DIR
    / "external"
    / "ChangeFormer"
)

CHECKPOINT_ROOT = (
    CHANGEFORMER_DIR
    / "checkpoints"
)

PROJECT_NAME = (
    "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_"
    "linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_"
    "embed_dim_256"
)

CHECKPOINT_NAME = "best_ckpt.pt"

IMG_SIZE = 256


# ============================================================
# ENVIRONMENT
# ============================================================

def _get_changeformer_python() -> Path:
    """
    Dynamically discovers the Python executable across Windows and Unix environments.
    Prioritizes the project-wide Windows virtual environment (CUDA enabled),
    then local Windows/Unix virtualenvs, and falls back to sys.executable.
    """
    candidates = [
        # 1. Project unified Windows virtual environment (CUDA-enabled)
        BACKEND_DIR.parents[1] / "Model Training" / "venv" / "Scripts" / "python.exe",
        # 2. Local Windows venvs if present
        BACKEND_DIR / "changeformer_env" / "Scripts" / "python.exe",
        BACKEND_DIR / ".venv" / "Scripts" / "python.exe",
        # 3. Unix venvs (macOS/Linux)
        BACKEND_DIR / "changeformer_env" / "bin" / "python",
        BACKEND_DIR / ".venv" / "bin" / "python",
        # 4. Current running process
        Path(sys.executable),
    ]
    for c in candidates:
        if c.exists():
            if sys.platform == "win32" and c.suffix.lower() != ".exe":
                continue
            return c
    return Path(sys.executable)


def _check_changeformer_environment() -> None:
    py_exe = _get_changeformer_python()
    if not py_exe.exists():
        raise RuntimeError(
            "ChangeFormer environment not found: "
            f"{py_exe}"
        )

    if not CHANGEFORMER_DIR.exists():
        raise RuntimeError(
            "ChangeFormer repository not found: "
            f"{CHANGEFORMER_DIR}"
        )

    checkpoint = (
        CHECKPOINT_ROOT
        / PROJECT_NAME
        / CHECKPOINT_NAME
    )

    if not checkpoint.exists():
        raise RuntimeError(
            "ChangeFormer checkpoint not found: "
            f"{checkpoint}"
        )


# ============================================================
# MAIN-PROCESS RGB PREPROCESSING
# ============================================================

def _robust_rgb(
    data: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Convert the first three bands of a raster array into
    an 8-bit RGB image using a robust 2-98 percentile stretch.
    """

    if data.ndim != 3:
        raise ValueError(
            f"Expected raster array (bands,H,W), got {data.shape}"
        )

    if data.shape[0] < 3:
        raise ValueError(
            "ChangeFormer requires at least 3 bands."
        )

    h, w = data.shape[1:]

    rgb = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for channel in range(3):
        band = np.asarray(
            data[channel],
            dtype=np.float32,
        )

        finite = (
            valid
            & np.isfinite(band)
        )

        if finite.sum() < 10:
            continue

        values = band[finite]

        low, high = np.percentile(
            values,
            [2, 98],
        )

        if (
            not np.isfinite(low)
            or not np.isfinite(high)
            or high <= low
        ):
            # Constant/degenerate band.
            scaled = np.zeros_like(
                band,
                dtype=np.uint8,
            )
        else:
            scaled_float = (
                (band - low)
                / (high - low)
            )

            scaled_float = np.clip(
                scaled_float,
                0.0,
                1.0,
            )

            scaled = (
                scaled_float * 255.0
            ).astype(np.uint8)

        rgb[:, :, channel] = scaled

    rgb[~valid] = 0

    return rgb


# ============================================================
# SUBPROCESS WORKER
# ============================================================

def _worker_load_model():
    """
    This function is executed ONLY inside changeformer_env.

    The main FastAPI environment never imports torch.
    """

    # Make the official repository importable.
    sys.path.insert(
        0,
        str(CHANGEFORMER_DIR),
    )

    import torch

    from models.basic_model import CDEvaluator

    checkpoint_dir = (
        CHECKPOINT_ROOT
        / PROJECT_NAME
    )

    use_cuda = torch.cuda.is_available()
    gpu_ids = [0] if use_cuda else []
    device = "cuda" if use_cuda else "cpu"

    args = SimpleNamespace(
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name=CHECKPOINT_NAME,

        gpu_ids=gpu_ids,

        device=device,

        project_name=PROJECT_NAME,

        n_class=2,
        embed_dim=256,
        net_G="ChangeFormerV6",

        img_size=IMG_SIZE,
        batch_size=1,

        dataset_mode="CD",
        phase="test",

        num_workers=0,

        data_name="quick_start_LEVIR",
        split="demo",

        output_folder=str(
            checkpoint_dir / "satquery_output"
        ),
    )

    model = CDEvaluator(args)

    model.load_checkpoint(
        CHECKPOINT_NAME
    )

    model.eval()

    return model, torch


def _worker_predict_tile(
    model,
    torch,
    before_tile: np.ndarray,
    after_tile: np.ndarray,
) -> np.ndarray:
    """
    Predict one 256x256 RGB pair.
    """

    if (
        before_tile.shape
        != after_tile.shape
    ):
        raise ValueError(
            "Before/after ChangeFormer tiles "
            "must have identical shapes."
        )

    if before_tile.shape != (
        IMG_SIZE,
        IMG_SIZE,
        3,
    ):
        raise ValueError(
            "ChangeFormer expects "
            f"{IMG_SIZE}x{IMG_SIZE}x3 tiles, "
            f"got {before_tile.shape}"
        )

    before_tensor = (
        torch.from_numpy(
            before_tile
            .astype(np.float32)
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    after_tensor = (
        torch.from_numpy(
            after_tile
            .astype(np.float32)
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    batch = {
        "A": before_tensor,
        "B": after_tensor,
        "name": ["satquery_tile.png"],
    }

    with torch.no_grad():
        score_map = model._forward_pass(
            batch
        )

    prediction = (
        score_map
        .detach()
        .cpu()
        .numpy()
    )

    # Official ChangeFormer _forward_pass returns
    # [B,1,H,W] visualization-style prediction.
    if prediction.ndim == 4:
        prediction = prediction[0]

    if prediction.ndim == 3:
        prediction = prediction[0]

    if prediction.ndim != 2:
        raise RuntimeError(
            "Unexpected ChangeFormer output shape: "
            f"{prediction.shape}"
        )

    return (
        prediction > 0
    ).astype(np.uint8)


def _worker_run(
    input_path: str,
    output_path: str,
) -> None:
    """
    Worker executed by changeformer_env.

    Input:
        NPZ containing before, after, valid.

    Output:
        NPY containing binary ChangeFormer mask.
    """

    arrays = np.load(
        input_path,
        allow_pickle=False,
    )

    before = arrays["before"]
    after = arrays["after"]
    valid = arrays["valid"].astype(bool)

    if before.shape != after.shape:
        raise ValueError(
            "Before/after arrays have different shapes: "
            f"{before.shape} vs {after.shape}"
        )

    if before.ndim != 3:
        raise ValueError(
            f"Expected (bands,H,W), got {before.shape}"
        )

    if before.shape[0] < 3:
        raise ValueError(
            "ChangeFormer requires at least 3 bands."
        )

    h, w = before.shape[1:]

    before_rgb = _robust_rgb(
        before,
        valid,
    )

    after_rgb = _robust_rgb(
        after,
        valid,
    )

    model, torch = _worker_load_model()

    output_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    tiles_processed = 0

    for y in range(
        0,
        h,
        IMG_SIZE,
    ):
        for x in range(
            0,
            w,
            IMG_SIZE,
        ):
            y2 = min(
                y + IMG_SIZE,
                h,
            )

            x2 = min(
                x + IMG_SIZE,
                w,
            )

            tile_h = y2 - y
            tile_w = x2 - x

            before_tile = np.zeros(
                (IMG_SIZE, IMG_SIZE, 3),
                dtype=np.uint8,
            )

            after_tile = np.zeros(
                (IMG_SIZE, IMG_SIZE, 3),
                dtype=np.uint8,
            )

            # Reflect padding where possible.
            before_crop = before_rgb[
                y:y2,
                x:x2,
            ]

            after_crop = after_rgb[
                y:y2,
                x:x2,
            ]

            if (
                tile_h == IMG_SIZE
                and tile_w == IMG_SIZE
            ):
                before_tile = before_crop
                after_tile = after_crop

            else:
                before_tile[
                    :tile_h,
                    :tile_w,
                ] = before_crop

                after_tile[
                    :tile_h,
                    :tile_w,
                ] = after_crop

                # Edge replication is safer than zeros
                # for ChangeFormer's convolution/attention
                # context.
                if tile_h < IMG_SIZE:
                    before_tile[
                        tile_h:,
                        :tile_w,
                    ] = before_tile[
                        tile_h - 1:tile_h,
                        :tile_w,
                    ]

                    after_tile[
                        tile_h:,
                        :tile_w,
                    ] = after_tile[
                        tile_h - 1:tile_h,
                        :tile_w,
                    ]

                if tile_w < IMG_SIZE:
                    before_tile[
                        :,
                        tile_w:,
                    ] = before_tile[
                        :,
                        tile_w - 1:tile_w,
                    ]

                    after_tile[
                        :,
                        tile_w:,
                    ] = after_tile[
                        :,
                        tile_w - 1:tile_w,
                    ]

            tile_prediction = _worker_predict_tile(
                model,
                torch,
                before_tile,
                after_tile,
            )

            output_mask[
                y:y2,
                x:x2,
            ] = tile_prediction[
                :tile_h,
                :tile_w,
            ]

            tiles_processed += 1

    output_mask[
        ~valid
    ] = 0

    np.save(
        output_path,
        output_mask,
    )

    changed_pixels = int(
        output_mask.sum()
    )

    valid_pixels = int(
        valid.sum()
    )

    fraction = (
        changed_pixels / valid_pixels
        if valid_pixels
        else 0.0
    )

    print(
        "CHANGEFORMER_WORKER_SUCCESS"
    )

    print(
        f"tiles_processed={tiles_processed}"
    )

    print(
        f"changed_pixels={changed_pixels}"
    )

    print(
        f"valid_pixels={valid_pixels}"
    )

    print(
        f"change_fraction={fraction:.6f}"
    )


# ============================================================
# PUBLIC ARRAY API
# ============================================================

def run_changeformer_arrays(
    first_data: np.ndarray,
    second_data: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, Any]:
    """
    Run ChangeFormer through the dedicated changeformer_env.

    This function is safe to call from the main FastAPI
    environment because it DOES NOT import torch there.
    """

    _check_changeformer_environment()

    first_data = np.asarray(
        first_data,
        dtype=np.float32,
    )

    second_data = np.asarray(
        second_data,
        dtype=np.float32,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    if first_data.shape != second_data.shape:
        raise ValueError(
            "Before/after arrays must have identical shapes: "
            f"{first_data.shape} != {second_data.shape}"
        )

    if first_data.ndim != 3:
        raise ValueError(
            "Expected raster arrays with shape "
            "(bands,height,width)."
        )

    if first_data.shape[0] < 3:
        raise ValueError(
            "ChangeFormer requires at least 3 bands."
        )

    if valid.shape != first_data.shape[1:]:
        raise ValueError(
            "Valid mask shape does not match raster dimensions: "
            f"{valid.shape} != {first_data.shape[1:]}"
        )

    with tempfile.TemporaryDirectory(
        prefix="satquery_changeformer_"
    ) as tmpdir:

        tmpdir = Path(tmpdir)

        input_path = (
            tmpdir / "input.npz"
        )

        output_path = (
            tmpdir / "output.npy"
        )

        np.savez_compressed(
            input_path,
            before=first_data,
            after=second_data,
            valid=valid.astype(np.uint8),
        )

        py_exe = _get_changeformer_python()

        command = [
            str(py_exe),
            str(Path(__file__).resolve()),
            "--worker",
            str(input_path),
            str(output_path),
        ]

        env = os.environ.copy()

        # Make sure the official repository is importable
        # inside the subprocess.
        env["PYTHONPATH"] = (
            str(CHANGEFORMER_DIR)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        process = subprocess.run(
            command,
            cwd=str(CHANGEFORMER_DIR),
            env=env,
            capture_output=True,
            text=True,
        )

        if process.returncode != 0:
            stderr = (
                process.stderr.strip()
                or process.stdout.strip()
                or "Unknown ChangeFormer worker error."
            )

            raise RuntimeError(
                "ChangeFormer subprocess failed: "
                + stderr[-4000:]
            )

        if not output_path.exists():
            raise RuntimeError(
                "ChangeFormer subprocess completed "
                "without producing a prediction."
            )

        mask = np.load(
            output_path,
            allow_pickle=False,
        )

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    if mask.shape != valid.shape:
        raise RuntimeError(
            "ChangeFormer returned invalid mask shape: "
            f"{mask.shape} != {valid.shape}"
        )

    mask = (
        (mask > 0)
        & valid
    )

    changed_pixels = int(
        mask.sum()
    )

    valid_pixels = int(
        valid.sum()
    )

    change_fraction = (
        changed_pixels / valid_pixels
        if valid_pixels
        else 0.0
    )

    return {
        "status": (
            "CHANGEFORMER_EXECUTED: "
            f"{'worker completed successfully'}; "
            f"change_fraction="
            f"{change_fraction:.4f}"
        ),
        "model": (
            "ChangeFormerV6 "
            "LEViR pretrained checkpoint"
        ),
        "mask": mask.astype(
            np.uint8
        ),
        "tiles_processed": (
            int(
                np.ceil(
                    first_data.shape[1]
                    / IMG_SIZE
                )
            )
            *
            int(
                np.ceil(
                    first_data.shape[2]
                    / IMG_SIZE
                )
            )
        ),
        "changed_pixels": changed_pixels,
        "valid_pixels": valid_pixels,
        "change_fraction": float(
            change_fraction
        ),
    }


# ============================================================
# PATH API
# ============================================================

def run_changeformer(
    before_path: str,
    after_path: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Kept for compatibility.

    SatQuery's actual pipeline should use
    run_changeformer_arrays() after spatial alignment.
    """

    raise RuntimeError(
        "run_changeformer() is intentionally disabled. "
        "ChangeFormer must run on the co-registered common "
        "analysis grid through run_changeformer_arrays()."
    )


# ============================================================
# CLI WORKER ENTRYPOINT
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        action="store_true",
    )

    parser.add_argument(
        "input_path",
        nargs="?",
    )

    parser.add_argument(
        "output_path",
        nargs="?",
    )

    args = parser.parse_args()

    if not args.worker:
        raise SystemExit(
            "This module is normally called by SatQuery."
        )

    if not args.input_path:
        raise SystemExit(
            "Missing worker input path."
        )

    if not args.output_path:
        raise SystemExit(
            "Missing worker output path."
        )

    _worker_run(
        args.input_path,
        args.output_path,
    )


if __name__ == "__main__":
    main()
