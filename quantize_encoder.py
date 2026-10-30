"""Dynamic-quantize the streaming zipformer encoder to int8 (encoder only).

This is a one-off offline step. It shrinks encoder.onnx ~4x and speeds up
CPU inference, usually with only a small WER change. The decoder/joiner are
left untouched (they are tiny and more sensitive to quantization).

Usage:
    pip install onnx onnxruntime
    python quantize_encoder.py

Then point the server at the result:
    set ENCODER_FILE=encoder.int8.onnx   # Windows (PowerShell: $env:ENCODER_FILE)
    export ENCODER_FILE=encoder.int8.onnx  # Linux/macOS

Always A/B test WER on a few reference clips before using in production.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "models" / "zipformer" / "encoder.onnx"
DEFAULT_OUTPUT = BASE_DIR / "models" / "zipformer" / "encoder.int8.onnx"


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="FP32 encoder.onnx")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="int8 output path")
    args = parser.parse_args()

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print(
            "Missing quantization tools. Install them with:\n"
            "    pip install onnx onnxruntime",
            file=sys.stderr,
        )
        return 1

    if not args.input.exists():
        print(f"Encoder not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Quantizing {args.input} ({_mb(args.input):.1f} MB) -> {args.output}")
    # Quantize only MatMul: safest for transformer/zipformer encoders and what
    # the official sherpa-onnx int8 exports do.
    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )
    print(f"Done: {args.output} ({_mb(args.output):.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
