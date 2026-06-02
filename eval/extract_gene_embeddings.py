#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract gene embeddings only.

Input:
- one h5ad file
  Rows are spots/cells; columns are genes.

Output:
- one .npy file: gene_emb_3584.npy by default
  Shape: [N, 3584] if GENE_EMB_TYPE = "qformer_3584"
  Shape: [N, 512]  if GENE_EMB_TYPE = "nicheformer_512"

Example:
CUDA_VISIBLE_DEVICES=0 python extract_gene_emb.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

import torch
from transformers import AutoModel, AutoProcessor
from peft import PeftModel


# =========================
# Path config: replace with your own paths
# =========================
MODEL_PATH = Path("/path/to/SciCore-Omics-or-local-model")
LORA_PATH = Path("/path/to/lora/checkpoint")  # set to None if not using LoRA

H5AD_PATH = Path("/path/to/input/gene_matrix.h5ad")
REFERENCE_H5AD_PATH = Path("/path/to/gene_tokenizer/model-symbel.h5ad")

OUT_EMB_PATH = Path("/path/to/output/gene_emb_3584.npy")


# =========================
# Runtime config
# =========================
DEVICE = "cuda:0"
DTYPE = "bfloat16"
SEED = 42
MAX_CELLS = None  # set to an int for debugging, e.g. 50

# Default output is the final gene representation projected into LLM hidden space.
# Use "nicheformer_512" if you only want the raw NicheFormer pooled embedding.
GENE_EMB_TYPE = "qformer_3584"  # "qformer_3584" or "nicheformer_512"

PROMPT_GENE = "(<gene>./</gene>)"


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


def load_reference_var_names(path: Path):
    ref_adata = ad.read_h5ad(path)
    return ref_adata.var_names.astype(str).tolist()


def build_aligned_gene_adata(one_gene: ad.AnnData, ref_var_names):
    """
    Align input genes to the gene_tokenizer reference gene order.
    Missing reference genes are filled with zero.
    Extra input genes are ignored.
    """
    ref_var_names = list(map(str, ref_var_names))
    src_var_names = list(map(str, one_gene.var_names))
    src_index = {g: i for i, g in enumerate(src_var_names)}
    n_ref = len(ref_var_names)

    X_src = one_gene.X

    if sp.issparse(X_src):
        X_src = X_src.tocsr()
        rows, cols, vals = [], [], []
        for j, g in enumerate(ref_var_names):
            if g in src_index:
                src_j = src_index[g]
                v = X_src[0, src_j]
                if v != 0:
                    rows.append(0)
                    cols.append(j)
                    vals.append(float(v))
        X_new = sp.csr_matrix((vals, (rows, cols)), shape=(1, n_ref), dtype=np.float32)
    else:
        X_src = np.asarray(X_src)
        X_new = np.zeros((1, n_ref), dtype=np.float32)
        for j, g in enumerate(ref_var_names):
            if g in src_index:
                X_new[0, j] = float(X_src[0, src_index[g]])

    new_obs = one_gene.obs.copy()
    new_var = pd.DataFrame(index=pd.Index(ref_var_names, name=one_gene.var_names.name))
    return ad.AnnData(X=X_new, obs=new_obs, var=new_var)


def unwrap_model(model):
    """
    Return the underlying model so we can directly access:
    nicheformer / gene_qformer / gene_projector.
    """
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


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
    return unwrap_model(model), processor


@torch.inference_mode()
def get_gene_embedding(model, processor, one_gene: ad.AnnData):
    prompts = [PROMPT_GENE]
    images = [[]]
    genes = [[one_gene]]

    inputs = processor(
        prompts,
        images,
        genes,
        return_tensors="pt",
    ).to(DEVICE)

    nf_output = model.nicheformer(
        input_ids=inputs["gene_input_ids"],
        attention_mask=inputs.get("gene_attention_mask", None),
    )  # [1, 1500, 512]

    if GENE_EMB_TYPE == "nicheformer_512":
        emb = nf_output.mean(dim=1)[0].cpu().float().numpy()  # [512]
        return emb

    if GENE_EMB_TYPE != "qformer_3584":
        raise ValueError(f"Unsupported GENE_EMB_TYPE: {GENE_EMB_TYPE}")

    # Same logic as the original script: remove the first 3 context tokens,
    # send NicheFormer tokens through Gene Q-Former and Gene Projector,
    # then mean-pool the projected 3584-d tokens.
    gene_tokens = nf_output[:, 3:, :]  # [1, 1497, 512]

    gene_attention_mask = inputs.get("gene_attention_mask", None)
    gene_pad_mask = None
    if gene_attention_mask is not None:
        gene_pad_mask = gene_attention_mask[:, 3:] == 0

    qformer_dtype = next(model.gene_qformer.parameters()).dtype
    projector_dtype = next(model.gene_projector.parameters()).dtype

    q_tokens = model.gene_qformer(
        gene_tokens.to(qformer_dtype),
        gene_pad_mask=gene_pad_mask,
    )  # [1, 32, 768]

    gene_tokens_3584 = model.gene_projector(
        q_tokens.to(projector_dtype),
    )  # [1, 32, 3584]

    emb = gene_tokens_3584.mean(dim=1)[0].cpu().float().numpy()  # [3584]
    return emb


def main():
    set_seed(SEED)

    if DEVICE.startswith("cuda"):
        torch.cuda.set_device(int(DEVICE.split(":")[1]))

    OUT_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading h5ad and reference genes...")
    adata = ad.read_h5ad(H5AD_PATH)
    ref_var_names = load_reference_var_names(REFERENCE_H5AD_PATH)

    if MAX_CELLS is not None:
        adata = adata[:MAX_CELLS].copy()

    print(f"[INFO] n_obs = {adata.n_obs}, n_vars = {adata.n_vars}")
    print(f"[INFO] n_reference_genes = {len(ref_var_names)}")

    print("[2/3] Loading model...")
    model, processor = load_model_and_processor()

    print("[3/3] Extracting gene embeddings...")
    gene_embs = []

    for i in range(adata.n_obs):
        raw_one_gene = adata[i:i + 1].copy()
        one_gene = build_aligned_gene_adata(raw_one_gene, ref_var_names)

        emb = get_gene_embedding(model, processor, one_gene)
        gene_embs.append(emb)

        if (i + 1) % 20 == 0 or (i + 1) == adata.n_obs:
            print(f"[INFO] done {i + 1}/{adata.n_obs}")

    gene_embs = np.stack(gene_embs, axis=0)
    np.save(OUT_EMB_PATH, gene_embs)

    print("[INFO] gene_embs.shape =", gene_embs.shape)
    print(f"[Saved] {OUT_EMB_PATH}")


if __name__ == "__main__":
    main()
