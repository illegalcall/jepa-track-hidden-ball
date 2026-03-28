#!/usr/bin/env python3
"""
Run a trained JEPA world model with an auxiliary state head on a folder of frames
or directly on a shell-game contact sheet.

This is a lightweight single-episode inference path for shell-game demos.
It assumes frames are numbered in order (000.png, 001.png, ...), and predicts
the ball's cup for each next frame after the history window.

Example:
    python demo_jepawm_predict.py \
      --checkpoint /path/to/lewm_auxonly_s2_h12_epoch_12_object.ckpt \
      --sheet demo_cases/case_1/sheet.png \
      --history-size 12 \
      --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from demo_sheet_to_frames import FULLRES_COLS, slice_sheet


class StateHead(nn.Module):
    # Compatibility shim for checkpoints saved when StateHead lived under __main__.
    def __init__(self, input_dim=192, hidden_dim=64, classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, x):
        return self.net(x)


def _resolve_path(obj, path: str):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur[key]
        else:
            cur = getattr(cur, key)
    return cur


def infer_positive_int(model, paths, default=None):
    for path in paths:
        try:
            value = _resolve_path(model, path)
        except Exception:
            continue
        if value is None:
            continue
        try:
            value = int(value)
        except Exception:
            continue
        if value > 0:
            return value
    return default


def load_frames(frames_dir: Path) -> np.ndarray:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    frames = [np.array(Image.open(path).convert("RGB"), dtype=np.uint8) for path in frame_paths]
    return np.stack(frames), frame_paths


def load_sheet(sheet_path: Path, cols: int = FULLRES_COLS):
    image = Image.open(sheet_path).convert("RGB")
    tiles, metadata = slice_sheet(image, cols=cols)
    if not tiles:
        raise ValueError(f"No frames were extracted from {sheet_path}")
    frames = [np.array(tile.convert("RGB"), dtype=np.uint8) for tile in tiles]
    frame_labels = [f"{idx:03d}.png" for idx in range(len(frames))]
    return np.stack(frames), frame_labels, metadata


@torch.no_grad()
def predict_episode(model, frames: np.ndarray, device: torch.device, history_size: int, action_dim: int):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def preprocess(frame: np.ndarray):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return (tensor - mean) / std

    tensors = torch.stack([preprocess(frame) for frame in frames]).to(device)
    enc_out = model.encoder(tensors, interpolate_pos_encoding=True)
    embs = model.projector(enc_out.last_hidden_state[:, 0])

    if len(frames) < history_size + 1:
        raise ValueError(
            f"Need at least history_size + 1 frames. "
            f"Got {len(frames)} with history_size={history_size}."
        )

    zero_actions = torch.zeros((1, history_size, action_dim), dtype=torch.float32, device=device)
    timeline = []

    for pred_idx in range(history_size, len(frames)):
        start = pred_idx - history_size
        hist_embs = embs[start:pred_idx].unsqueeze(0)
        act_embs = model.action_encoder(zero_actions)
        pred_out = model.predictor(hist_embs, act_embs)
        pred_emb = pred_out[:, -1, :]
        logits = model.state_head(pred_emb)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        timeline.append({
            "frame_index": pred_idx,
            "predicted_cup": int(np.argmax(probs)) + 1,
            "cup_probs": [float(p) for p in probs],
        })

    return timeline


def main():
    parser = argparse.ArgumentParser(description="Run JEPA shell-game inference on a folder of frames")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--sheet-cols", type=int, default=FULLRES_COLS)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if bool(args.frames_dir) == bool(args.sheet):
        raise ValueError("Pass exactly one of --frames-dir or --sheet.")

    # Make sure the original training modules are importable when unpickling.
    le_wm_dir = Path(__file__).resolve().parent / "le-wm"
    if le_wm_dir.exists():
        sys.path.insert(0, str(le_wm_dir))

    device = torch.device(args.device)

    if args.frames_dir:
        frames, frame_paths = load_frames(Path(args.frames_dir))
        input_info = {
            "kind": "frames_dir",
            "path": str(Path(args.frames_dir).resolve()),
        }
    else:
        frames, frame_paths, sheet_meta = load_sheet(Path(args.sheet), cols=args.sheet_cols)
        input_info = {
            "kind": "sheet",
            "path": str(Path(args.sheet).resolve()),
            "cols": args.sheet_cols,
            "sheet_metadata": sheet_meta,
        }
    model = torch.load(args.checkpoint, map_location=device, weights_only=False).to(device)
    model.eval()

    history_size = args.history_size
    if history_size is None:
        history_size = infer_positive_int(model, [
            "hparams.wm.history_size",
            "hparams.history_size",
            "cfg.wm.history_size",
            "cfg.history_size",
            "config.wm.history_size",
            "config.history_size",
            "wm.history_size",
        ])
    if history_size is None:
        raise ValueError("Could not infer history size from checkpoint. Pass --history-size.")

    timeline = predict_episode(
        model=model,
        frames=frames,
        device=device,
        history_size=history_size,
        action_dim=args.action_dim,
    )

    result = {
        "checkpoint": args.checkpoint,
        "input": input_info,
        "num_frames": len(frame_paths),
        "history_size": history_size,
        "final_predicted_cup": timeline[-1]["predicted_cup"],
        "final_cup_probs": timeline[-1]["cup_probs"],
        "timeline": timeline,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"Saved {output_path}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
