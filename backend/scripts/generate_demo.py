"""Create a tiny deterministic image collection for local smoke testing."""
import argparse
from pathlib import Path
import cv2
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/images/objects"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    colors = [(30,30,220),(30,200,30),(220,40,30),(30,210,220),(200,40,180),(180,180,30)]
    labels = ["red", "green", "blue", "yellow", "purple", "cyan"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        for variant in range(3):
            image = np.zeros((160,160,3), dtype=np.uint8)
            shade = tuple(max(0, min(255, channel - variant * 12)) for channel in color)
            image[:] = shade
            radius = 25 + variant * 9
            cv2.circle(image, (80,80), radius, (255,255,255), 2 + variant)
            cv2.line(image, (20 + variant * 10, 20), (140, 140 - variant * 8), (0,0,0), 2)
            cv2.imwrite(str(args.output / f"{label}_{variant}.jpg"), image)

if __name__ == "__main__": main()
