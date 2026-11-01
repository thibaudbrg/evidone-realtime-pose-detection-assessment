"""Export the pose model to another inference runtime."""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO pose model to ONNX or CoreML.")
    parser.add_argument("--model", default="yolo26n-pose.pt",
                        help="path to the YOLO pose model (default: yolo26n-pose.pt)")
    parser.add_argument("--format", choices=("onnx", "coreml"), required=True,
                        help="target runtime format")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="export input image size (default: 640)")
    parser.add_argument("--output", type=Path,
                        help="optional expected output path, checked after export")
    args = parser.parse_args()
    if args.imgsz < 32:
        parser.error(f"--imgsz must be at least 32, got {args.imgsz}")
    return args


def main() -> None:
    args = parse_args()
    if not Path(args.model).exists():
        sys.exit(f"Error: model file not found: {args.model}")
    try:
        model = YOLO(args.model)
        if model.task != "pose":
            sys.exit(f"Error: {args.model} is a '{model.task}' model, expected a pose model.")
        exported = model.export(format=args.format, imgsz=args.imgsz)
    except Exception as exc:
        sys.exit(f"Error: export to {args.format} failed: {exc}")

    exported_path = Path(str(exported))
    if args.output and exported_path.resolve() != args.output.resolve():
        sys.exit(f"Error: export completed at {exported_path}, not the requested path {args.output}.")
    print(f"Exported {args.model} to {exported_path}")


if __name__ == "__main__":
    main()
