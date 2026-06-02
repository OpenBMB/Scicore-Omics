#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract image embeddings only.

Input:
- one image file, or
- a directory containing image patches/files

Output:
- one .npy file: image_emb_3584.npy
  Shape: [N, 3584]
  Order: sorted by image filename if INPUT_IMAGE_PATH is a directory.

Example:
CUDA_VISIBLE_DEVICES=0 python extract_image_emb.py
"""

from pathlib import Path
import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor
from peft import PeftModel


# =========================
# Path config: replace with your own paths
# =========================
MODEL_PATH = Path("/path/to/SciCore-Omics-or-local-model")
LORA_PATH = Path("/path/to/lora/checkpoint")  # set to None if not using LoRA

INPUT_IMAGE_PATH = Path("/path/to/input/images_or_one_image")
OUT_EMB_PATH = Path("/path/to/output/image_emb_3584.npy")


# =========================
# Runtime config
# =========================
DEVICE = "cuda:0"
DTYPE = "bfloat16"
SEED = 42

PROMPT_IMG = "图像 (<image>./</image>)"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def list_image_files(path: Path):
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image suffix: {path}")
        return [path]

    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if len(files) == 0:
            raise FileNotFoundError(f"No image files found in: {path}")
        return files

    raise FileNotFoundError(f"INPUT_IMAGE_PATH does not exist: {path}")


def load_model_and_processor():
    torch_dtype = get_torch_dtype(DTYPE)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    base_model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(DEVICE)

    if LORA_PATH is not None:
        print(f"[INFO] loading LoRA from: {LORA_PATH}")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
    else:
        model = base_model

    model.eval()
    return model, processor


def call_get_vllm_embedding(model, inputs):
    """
    Compatible with both a plain model and a PEFT-wrapped model.
    """
    if hasattr(model, "get_vllm_embedding"):
        return model.get_vllm_embedding(inputs)

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.get_vllm_embedding(inputs)

    raise AttributeError("Cannot find get_vllm_embedding() on model.")


@torch.inference_mode()
def get_image_embedding(model, processor, image: Image.Image):
    """
    Extract image tokens from inputs_embeds according to image_bound,
    then mean-pool them into one 3584-d vector.
    """
    prompts = [PROMPT_IMG]
    images = [[image]]
    genes = [[]]

    inputs = processor(
        prompts,
        images,
        genes,
        max_slice_nums=1,
        return_tensors="pt",
    ).to(DEVICE)

    inputs_embeds, _ = call_get_vllm_embedding(model, inputs)
    cur = inputs_embeds[0]  # [L, 3584]

    if "image_bound" not in inputs:
        raise ValueError("image_bound not found in processor outputs.")

    image_bounds = inputs["image_bound"][0]
    if image_bounds.numel() == 0:
        raise ValueError("image_bound is empty.")

    image_tokens = torch.cat([cur[s:e] for s, e in image_bounds.tolist()], dim=0)
    emb = image_tokens.mean(dim=0).cpu().float().numpy()  # [3584]
    return emb


def main():
    set_seed(SEED)

    if DEVICE.startswith("cuda"):
        torch.cuda.set_device(int(DEVICE.split(":")[1]))

    OUT_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(INPUT_IMAGE_PATH)
    print(f"[INFO] n_images = {len(image_files)}")

    model, processor = load_model_and_processor()

    image_embs = []
    for i, img_path in enumerate(image_files, 1):
        image = Image.open(img_path).convert("RGB")
        emb = get_image_embedding(model, processor, image)
        image_embs.append(emb)

        if i % 20 == 0 or i == len(image_files):
            print(f"[INFO] done {i}/{len(image_files)}")

    image_embs = np.stack(image_embs, axis=0)  # [N, 3584]
    np.save(OUT_EMB_PATH, image_embs)

    print("[INFO] image_embs.shape =", image_embs.shape)
    print(f"[Saved] {OUT_EMB_PATH}")


if __name__ == "__main__":
    main()
