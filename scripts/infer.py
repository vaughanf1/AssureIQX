#!/usr/bin/env python3
"""Run single-image or batch inference with Grad-CAM overlay.

Accepts either a single ``--image`` path or a directory via
``--input-dir`` and produces predicted class labels with softmax
confidence scores and optional Grad-CAM overlay images.

Single-image mode:
    python scripts/infer.py --image path/to/xray.jpeg --checkpoint checkpoints/best_stratified.pt

Batch mode:
    python scripts/infer.py --input-dir path/to/images/ --checkpoint checkpoints/best_stratified.pt

Options:
    --output-dir    Directory for Grad-CAM overlays and batch results (default: results/inference/)
    --no-gradcam    Skip Grad-CAM overlay generation for faster inference
    --config        YAML config file (default: configs/default.yaml)
    --override      Config overrides in key.subkey=value format

Implemented in: Phase 6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_test_transforms  # noqa: E402
from src.explainability.gradcam import (  # noqa: E402
    create_overlay,
    denormalize_tensor,
    generate_gradcam,
)
from src.models.factory import create_model, get_device, load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

# Supported image extensions for globbing
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _load_model(checkpoint_path: str, cfg: dict, device: torch.device):
    """Load model from checkpoint and return (model, class_names).

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        cfg: Full config dict (fallback for model creation).
        device: Target device.

    Returns:
        Tuple of (model, class_names) with model in eval mode on device.
    """
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    ckpt_config = ckpt["config"]
    class_names = ckpt["class_names"]

    model = create_model(ckpt_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        "Model loaded: %s (epoch %d, val_loss=%.6f)",
        ckpt_config["model"]["backbone"],
        ckpt["epoch"],
        ckpt["val_loss"],
    )
    return model, class_names


def _preprocess_image(
    image_path: Path, transform
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file.
        transform: Albumentations transform pipeline.

    Returns:
        Tuple of (input_tensor, rgb_array) where input_tensor has shape
        (1, C, H, W) and rgb_array is the original image as numpy HWC uint8.
    """
    pil_img = Image.open(image_path).convert("RGB")
    rgb_array = np.array(pil_img)

    transformed = transform(image=rgb_array)
    tensor = transformed["image"]  # (C, H, W) float32
    input_tensor = tensor.unsqueeze(0)  # (1, C, H, W)

    return input_tensor, rgb_array


def _run_single_inference(
    image_path: Path,
    model: torch.nn.Module,
    class_names: list[str],
    transform,
    device: torch.device,
    output_dir: Path,
    skip_gradcam: bool = False,
) -> dict:
    """Run inference on a single image.

    Performs forward pass, computes softmax confidences, generates
    Grad-CAM overlay (unless skipped), saves overlay PNG, and prints
    results to console.

    Args:
        image_path: Path to input image.
        model: Trained model in eval mode on device.
        class_names: List of class name strings.
        transform: Albumentations test transform pipeline.
        device: Torch device.
        output_dir: Directory to save Grad-CAM overlays.
        skip_gradcam: If True, skip Grad-CAM generation.

    Returns:
        Dict with image_name, prediction, confidence, and per-class scores.
    """
    input_tensor, _ = _preprocess_image(image_path, transform)
    input_tensor = input_tensor.to(device)

    # Forward pass with no_grad for prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    probs_np = probs.cpu().numpy()[0]  # (num_classes,)
    pred_idx = int(probs_np.argmax())
    pred_class = class_names[pred_idx]
    pred_confidence = float(probs_np[pred_idx])

    # Build per-class scores dict
    scores = {name: float(probs_np[i]) for i, name in enumerate(class_names)}

    # Grad-CAM overlay
    overlay_path = None
    if not skip_gradcam:
        # Grad-CAM needs gradients -- do NOT wrap in torch.no_grad()
        heatmap = generate_gradcam(model, input_tensor, pred_idx)

        # Denormalize the input tensor for overlay
        rgb_img = denormalize_tensor(input_tensor.squeeze(0))
        overlay = create_overlay(rgb_img, heatmap)

        # Save overlay as PNG using PIL
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_filename = f"{image_path.stem}_gradcam.png"
        overlay_path = output_dir / overlay_filename
        Image.fromarray(overlay).save(overlay_path)

    # Print results to console
    print(f"Image: {image_path.name}")
    print(f"Prediction: {pred_class} (confidence: {pred_confidence:.3f})")
    scores_str = ", ".join(f"{name}={scores[name]:.3f}" for name in class_names)
    print(f"Scores: {scores_str}")
    if overlay_path is not None:
        print(f"Grad-CAM overlay: {overlay_path}")
    elif skip_gradcam:
        print("Grad-CAM overlay: skipped (--no-gradcam)")
    print()

    return {
        "image_name": image_path.name,
        "prediction": pred_class,
        "confidence": pred_confidence,
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-image or batch inference with Grad-CAM overlay."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key.subkey=value format",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image for inference",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to directory of images for batch inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint .pt file (default: from config inference.default_checkpoint)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for Grad-CAM overlays and batch results (default: results/inference/)",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM overlay generation for faster inference",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ── Argument validation ──────────────────────────────────────
    if args.image is None and args.input_dir is None:
        parser.error(
            "Error: exactly one of --image or --input-dir must be provided."
        )
    if args.image is not None and args.input_dir is not None:
        parser.error(
            "Error: provide only one of --image or --input-dir, not both."
        )

    # ── Load config ──────────────────────────────────────────────
    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.get("seed", 42))

    # ── Resolve checkpoint path ──────────────────────────────────
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = PROJECT_ROOT / cfg["inference"]["default_checkpoint"]

    if not checkpoint_path.exists():
        print(
            f"Error: checkpoint not found at '{checkpoint_path}'.\n"
            f"Provide a valid path via --checkpoint or set "
            f"inference.default_checkpoint in your config.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Resolve output directory ─────────────────────────────────
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device and model ─────────────────────────────────────────
    device = get_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    model, class_names = _load_model(str(checkpoint_path), cfg, device)
    logger.info("Class names: %s", class_names)

    # ── Transform pipeline (deterministic, no augmentation) ──────
    image_size = cfg["data"]["image_size"]
    transform = get_test_transforms(image_size)

    # ── Single-image mode ────────────────────────────────────────
    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.exists():
            print(
                f"Error: image file not found: '{image_path}'",
                file=sys.stderr,
            )
            sys.exit(1)

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"Warning: unsupported image extension '{image_path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                file=sys.stderr,
            )

        logger.info("Single-image inference: %s", image_path)
        _run_single_inference(
            image_path=image_path,
            model=model,
            class_names=class_names,
            transform=transform,
            device=device,
            output_dir=output_dir,
            skip_gradcam=args.no_gradcam,
        )

    # ── Batch mode ───────────────────────────────────────────────
    elif args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(
                f"Error: input directory not found: '{input_dir}'",
                file=sys.stderr,
            )
            sys.exit(1)

        # Collect all image files with supported extensions
        image_files = sorted(
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_files:
            print(
                f"Error: no image files found in '{input_dir}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                file=sys.stderr,
            )
            sys.exit(1)

        logger.info(
            "Batch inference: %d images in %s", len(image_files), input_dir
        )
        print(f"Found {len(image_files)} images in {input_dir}\n")

        results = []
        for i, image_path in enumerate(image_files, 1):
            logger.info("Processing image %d/%d: %s", i, len(image_files), image_path.name)
            result = _run_single_inference(
                image_path=image_path,
                model=model,
                class_names=class_names,
                transform=transform,
                device=device,
                output_dir=output_dir,
                skip_gradcam=args.no_gradcam,
            )
            results.append(result)

        # Save batch results JSON
        batch_results_path = output_dir / "batch_results.json"
        with open(batch_results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved batch results: %s", batch_results_path)

        # Print summary table
        print("=" * 70)
        print("BATCH INFERENCE SUMMARY")
        print("=" * 70)
        print(f"{'Image':<30s} {'Prediction':<15s} {'Confidence':>10s}")
        print("-" * 55)
        for r in results:
            print(
                f"{r['image_name']:<30s} {r['prediction']:<15s} {r['confidence']:>10.3f}"
            )
        print("=" * 70)
        print(f"Total: {len(results)} images processed")
        print(f"Results saved: {batch_results_path}")
        if not args.no_gradcam:
            print(f"Overlays saved: {output_dir}/")
        print()


if __name__ == "__main__":
    main()
