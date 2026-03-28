#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_SIZE = 224
FULLRES_COLS = 5
FULLRES_ROW_PITCH = 242
BLANK_STD_THRESHOLD = 8.0
BLANK_MEAN_THRESHOLD = 236.0


def trim_light_border(image: Image.Image, threshold: int = 248):
    gray = np.array(image.convert("L"))
    mask = gray < threshold
    if not mask.any():
        return image, [0, 0, image.width, image.height]

    ys, xs = np.where(mask)
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return image.crop((left, top, right, bottom)), [left, top, right, bottom]


def is_blank(tile: Image.Image) -> bool:
    arr = np.asarray(tile.convert("RGB"), dtype=np.float32)
    return float(arr.std()) < BLANK_STD_THRESHOLD and float(arr.mean()) > BLANK_MEAN_THRESHOLD


def slice_sheet(image: Image.Image, cols: int):
    trimmed, trim_box = trim_light_border(image)
    source_tile = trimmed.width / cols
    source_row_pitch = source_tile * (FULLRES_ROW_PITCH / FRAME_SIZE)

    tiles = []
    row = 0
    while True:
        top = row * source_row_pitch
        bottom = top + source_tile
        if bottom > trimmed.height + max(2.0, source_tile * 0.12):
            break

        for col in range(cols):
            left = col * source_tile
            right = left + source_tile
            if right > trimmed.width + max(2.0, source_tile * 0.08):
                continue

            box = (
                int(round(left)),
                int(round(top)),
                int(round(right)),
                int(round(bottom)),
            )
            tile = trimmed.crop(box).resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS)
            if is_blank(tile):
                continue
            tiles.append(tile)
        row += 1

    metadata = {
        "input_size": [image.width, image.height],
        "trimmed_size": [trimmed.width, trimmed.height],
        "trim_box": trim_box,
        "cols": cols,
        "source_tile_size": source_tile,
        "source_row_pitch": source_row_pitch,
        "num_frames": len(tiles),
        "row_count_estimate": row,
    }
    return tiles, metadata


def main():
    parser = argparse.ArgumentParser(description="Slice a 5-column shell-game contact sheet into numbered frames")
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cols", type=int, default=FULLRES_COLS)
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(sheet_path).convert("RGB")
    tiles, metadata = slice_sheet(image, cols=args.cols)
    if not tiles:
        raise ValueError("No frames were extracted from the uploaded sheet.")

    for idx, tile in enumerate(tiles):
        tile.save(output_dir / f"{idx:03d}.png")

    manifest = {
        "sheet_path": str(sheet_path.resolve()),
        "frames_dir": str(output_dir.resolve()),
        **metadata,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
