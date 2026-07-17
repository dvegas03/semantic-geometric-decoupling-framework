# One end-to-end open-vocabulary grounding query on an RGB-D frame.

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np

from engine_grounder.agent import GroundingAgent
from engine_grounder.perception.vlm_agent import VLMAgent
from engine_grounder.spatial.lifting import BBoxLifter
from engine_grounder.utils.synthetic_data import SyntheticDataGenerator


def load_frame(args):
    if args.synthetic:
        generator = SyntheticDataGenerator(size=(240, 320), true_depth=2.0)
        depth = generator.generate_noisy_crop(void_ratio=args.void_ratio, seed=0).astype(np.float32)
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[60:180, 100:220] = (200, 60, 60)
        intrinsics = (500.0, 500.0, 160.0, 120.0)
        return rgb, depth, intrinsics
    rgb = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
    depth = np.load(args.depth).astype(np.float32)
    return rgb, depth, tuple(args.intrinsics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", help="Path to RGB image (PNG/JPG)")
    parser.add_argument("--depth", help="Path to metric depth map (.npy, metres)")
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--query", default="the red square")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--void-ratio", type=float, default=0.5)
    args = parser.parse_args()
    if not args.synthetic and not (args.rgb and args.depth and args.intrinsics):
        parser.error("provide --rgb/--depth/--intrinsics, or use --synthetic")

    rgb, depth, intrinsics = load_frame(args)
    agent = GroundingAgent(vlm=VLMAgent(model_name=args.model), lifter=BBoxLifter())
    response = agent.ground(rgb, depth, intrinsics, args.query)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
