#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import sys
import json
import glob
import argparse
from typing import List, Dict, Tuple
from PIL import Image
from tqdm import tqdm

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util


def resolve_path(root: str, maybe_path: str) -> str:
    """If maybe_path is absolute or exists, return it; else join with root."""
    if not maybe_path:
        return ""
    if os.path.isabs(maybe_path) and os.path.exists(maybe_path):
        return maybe_path
    p = os.path.join(root, maybe_path)
    return p if os.path.exists(p) else maybe_path  # return as-is if still missing


def list_images_by_stem(folder: str) -> Dict[str, str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    stem_to_path = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        stem_to_path[stem] = p
    return stem_to_path


def load_pairs_from_csv(csv_path: str, fg_dir: str, bg_dir: str) -> List[Tuple[str, str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            _id = str(r.get("id", "")).strip()
            fg = resolve_path(fg_dir, str(r.get("fg", "")).strip())
            bg = resolve_path(bg_dir, str(r.get("bg", "")).strip())
            if not _id or not os.path.exists(fg) or not os.path.exists(bg):
                continue
            rows.append((_id, fg, bg))
    return rows


def load_pairs_from_json(json_path: str, fg_dir: str, bg_dir: str) -> List[Tuple[str, str, str]]:
    items = json.load(open(json_path, "r", encoding="utf-8"))
    if isinstance(items, dict) and "items" in items:
        items = items["items"]
    rows = []
    for it in items:
        _id = str(it.get("id", "")).strip()
        if not _id:
            continue
        # Try any extension
        fg_candidates = glob.glob(os.path.join(fg_dir, _id) + ".*")
        bg_candidates = glob.glob(os.path.join(bg_dir, _id) + ".*")
        if not fg_candidates or not bg_candidates:
            continue
        rows.append((_id, fg_candidates[0], bg_candidates[0]))
    return rows


def load_pairs_auto(fg_dir: str, bg_dir: str) -> List[Tuple[str, str, str]]:
    fg_map = list_images_by_stem(fg_dir)
    bg_map = list_images_by_stem(bg_dir)
    common = sorted(set(fg_map.keys()) & set(bg_map.keys()))
    return [(sid, fg_map[sid], bg_map[sid]) for sid in common]


def make_captioner(model_name: str, device: str = "cpu", dtype: str = "fp32"):
    dtype_map = {"fp32": torch.float32, "bfloat16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map.get(dtype.lower(), torch.float32)

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def caption_image(processor, model, image: Image.Image, device: str, max_new_tokens: int = 18) -> str:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return text


def main():
    ap = argparse.ArgumentParser(description="Lightweight semantic FG–BG checker (CPU)")
    ap.add_argument("--data", type=str, default=None,
                    help="Optional: CSV[id,fg,bg] or JSON list of {id}. If omitted, auto-pair by filename stems.")
    ap.add_argument("--fg_dir", type=str, required=True, help="Folder of foreground crops")
    ap.add_argument("--bg_dir", type=str, required=True, help="Folder of background crops")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--caption_model", type=str, default="Salesforce/blip-image-captioning-base",
                    help="BLIP caption model (base recommended for CPU)")
    ap.add_argument("--caption_device", type=str, default="cpu", choices=["cpu"], help="Caption device (CPU only here)")
    ap.add_argument("--caption_dtype", type=str, default="fp32", choices=["fp32", "bfloat16", "fp16"])
    ap.add_argument("--max_new_tokens", type=int, default=16, help="Shorter = faster")
    ap.add_argument("--sts_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model for similarity")
    ap.add_argument("--threshold", type=float, default=0.55, help="Decision threshold on cosine similarity (0..1)")
    ap.add_argument("--max_images", type=int, default=None, help="Optional cap on number of pairs")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Build pairs
    pairs: List[Tuple[str, str, str]] = []
    if args.data and args.data.lower().endswith(".csv"):
        pairs = load_pairs_from_csv(args.data, args.fg_dir, args.bg_dir)
    elif args.data and args.data.lower().endswith(".json"):
        pairs = load_pairs_from_json(args.data, args.fg_dir, args.bg_dir)
    else:
        pairs = load_pairs_auto(args.fg_dir, args.bg_dir)

    if args.max_images:
        pairs = pairs[: int(args.max_images)]

    if not pairs:
        print("[ERR] No (id, fg, bg) pairs found. Check --fg_dir/--bg_dir and --data format.")
        sys.exit(1)

    print(f"[info] pairs: {len(pairs)}")
    print(f"[info] caption model: {args.caption_model} (device={args.caption_device}, dtype={args.caption_dtype})")
    print(f"[info] sts model: {args.sts_model}")
    print(f"[info] writing: {args.out_csv}")

    # Models (CPU)
    processor, blip = make_captioner(args.caption_model, device="cpu", dtype=args.caption_dtype)
    st_model = SentenceTransformer(args.sts_model, device="cpu")

    # Process
    out_rows = []
    for _id, fg_path, bg_path in tqdm(pairs, desc="semantic"):
        try:
            fg_img = Image.open(fg_path).convert("RGB")
            bg_img = Image.open(bg_path).convert("RGB")

            # Small pre-resize helps CPU a bit (BLIP will still resize internally)
            fg_img = fg_img.resize((448, 448))
            bg_img = bg_img.resize((448, 448))

            fg_text = caption_image(processor, blip, fg_img, device="cpu", max_new_tokens=args.max_new_tokens)
            bg_text = caption_image(processor, blip, bg_img, device="cpu", max_new_tokens=args.max_new_tokens)

            # STS cosine similarity in [0,1]
            embs = st_model.encode([fg_text, bg_text], convert_to_tensor=True, normalize_embeddings=True)
            sim = float(util.cos_sim(embs[0], embs[1]).item())  # already in [-1,1] but near [0,1] due to norm
            # clamp to [0,1]
            sim01 = max(0.0, min(1.0, (sim + 1.0) / 2.0)) if sim < 0 else min(1.0, sim)

            label = "Match" if sim01 >= args.threshold else "Mismatch"

            out_rows.append({
                "id": _id,
                "fg_path": os.path.abspath(fg_path),
                "bg_path": os.path.abspath(bg_path),
                "fg_text": fg_text,
                "bg_text": bg_text,
                "sts01": f"{sim01:.4f}",
                "label": label
            })
        except Exception as ex:
            out_rows.append({
                "id": _id,
                "fg_path": fg_path,
                "bg_path": bg_path,
                "fg_text": "",
                "bg_text": "",
                "sts01": "",
                "label": f"ERROR: {ex}"
            })

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "fg_path", "bg_path", "fg_text", "bg_text", "sts01", "label"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"[done] wrote {len(out_rows)} rows → {args.out_csv}")
    print("[note] Tune --threshold (default 0.55). Lower it for more permissive matches, raise for stricter.")


if __name__ == "__main__":
    main()
