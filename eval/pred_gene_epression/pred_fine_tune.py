#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from peft import PeftModel

# ================================
# Paths & Hyperparams
# ================================
DATA_DIR = Path("/data/path/to/")
OUTPUT_DIR = Path("/fine_tune_output/path/to/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR = Path("/data/path/to/patches_hires_128")
MODEL_PATH = "/model/path/to/"
# path_to_adapter="/cpt_output/path/to/"


GENE_EXP = DATA_DIR / "heart_HEG.npy"
TRAIN_MASK = DATA_DIR / "train_mask.npy"
TEST_MASK  = DATA_DIR / "test_mask.npy"

DEVICE = "cuda:3"
BATCH_SIZE = 2              
EPOCHS = 30                
LR_HEAD = 1e-3
LR_VISION = 5e-6            
PROMPT = "图像 (<image>./</image>)"


# ================================
# Dataset
# ================================
class HeartImgGeneDataset(Dataset):
    def __init__(self, mask, gene_mat, patch_dir):
        self.indices = np.where(mask)[0]
        self.gene_mat = gene_mat
        self.patch_dir = Path(patch_dir)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ridx = self.indices[idx]
        img_path = self.patch_dir / f"spot_{ridx:04d}.png"
        img = Image.open(img_path).convert("RGB")
        y = torch.tensor(self.gene_mat[ridx], dtype=torch.float32)
        return img, y


# ================================
# Stronger Regression Head (Route A)
# ================================
class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = 2048
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ================================
# PCC
# ================================
def compute_gene_wise_pcc(pred, true):
    K = true.shape[1]
    pccs = []
    for i in range(K):
        if np.std(pred[:, i]) < 1e-6 or np.std(true[:, i]) < 1e-6:
            pccs.append(0)
        else:
            pccs.append(pearsonr(pred[:, i], true[:, i])[0])
    return np.array(pccs)


# ================================
# MAIN
# ================================
def main():

    # ---- Load Data ----
    Y = np.load(GENE_EXP)
    train_mask = np.load(TRAIN_MASK).astype(bool)
    test_mask  = np.load(TEST_MASK).astype(bool)

    N, K = Y.shape
    print(f"[INFO] Spots={N}, Genes={K}")
    print(f"[INFO] Train={train_mask.sum()}, Test={test_mask.sum()}")

    train_ds = HeartImgGeneDataset(train_mask, Y, PATCH_DIR)
    test_ds  = HeartImgGeneDataset(test_mask,  Y, PATCH_DIR)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x
    )

    # ---- Load Model & Processor ----
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE)
    
    # model = PeftModel.from_pretrained(
    #     model,
    #     path_to_adapter,
    #     device_map="auto",  
    #     trust_remote_code=True
    # )
    model = model.eval()

    EMB_DIM = 3584
    head = MLPProbe(EMB_DIM, K).to(DEVICE)

    # ---- Freeze all params ----
    for name, param in model.named_parameters():
        param.requires_grad = False

    # ---- Unfreeze vpm layers 20-26 + post_layernorm (Route A) ----
    vision_params = []
    for name, param in model.named_parameters():
        if name.startswith("vpm.encoder.layers.20") or \
           name.startswith("vpm.encoder.layers.21") or \
           name.startswith("vpm.encoder.layers.22") or \
           name.startswith("vpm.encoder.layers.23") or \
           name.startswith("vpm.encoder.layers.24") or \
           name.startswith("vpm.encoder.layers.25") or \
           name.startswith("vpm.encoder.layers.26") or \
           name.startswith("vpm.post_layernorm"):
            param.requires_grad = True
            vision_params.append(param)

    print(f"[INFO] Unfrozen {len(vision_params)} vision params (layers 20-26 + post_layernorm)")

    optimizer = torch.optim.Adam([
        {"params": head.parameters(),   "lr": LR_HEAD},
        {"params": vision_params,       "lr": LR_VISION},
    ])

    criterion = nn.MSELoss()

    # ---- Track metrics ----
    train_losses = []
    test_pccs = []
    test_mses = []

    best_pcc = -999

    print("\n===== Route A: Start Finetuning Vision (20-26) + Stronger Head =====")

    for epoch in range(EPOCHS):
        model.train()
        head.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            imgs, ys = zip(*batch)
            imgs = list(imgs)
            ys = torch.stack(ys, dim=0).to(DEVICE)

            B = len(imgs)
            prompts = [PROMPT] * B
            images  = [[img] for img in imgs]
            genes   = [[] for _ in range(B)]

            inputs = processor(
                prompts, images, genes,
                max_slice_nums=1,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, vis_states = model.get_vllm_embedding(inputs)
                vs = torch.cat(vis_states, dim=0)  # (B, 64, 3584)
                img_emb = vs.mean(dim=1)
                pred = head(img_emb)
                loss = criterion(pred.float(), ys.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B

        avg_train_loss = total_loss / len(train_ds)
        train_losses.append(avg_train_loss)

        # ---- Evaluation ----
        model.eval()
        head.eval()
        preds, gts = [], []

        with torch.no_grad():
            for batch in test_loader:
                imgs, ys = zip(*batch)
                imgs = list(imgs)
                ys_np = torch.stack(ys, dim=0).numpy()
                gts.append(ys_np)

                B = len(imgs)
                prompts = [PROMPT] * B
                images  = [[img] for img in imgs]
                genes   = [[] for _ in range(B)]

                inputs = processor(
                    prompts, images, genes,
                    max_slice_nums=1,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, vis_states = model.get_vllm_embedding(inputs)
                    vs = torch.cat(vis_states, dim=0)
                    img_emb = vs.mean(dim=1)
                    pred = head(img_emb).cpu().float().numpy()

                preds.append(pred)

        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        pcc_vec = compute_gene_wise_pcc(preds, gts)
        avg_pcc = pcc_vec.mean()
        avg_mse = mean_squared_error(gts, preds)

        test_pccs.append(avg_pcc)
        test_mses.append(avg_mse)

        print(f"[Epoch {epoch+1}/{EPOCHS}] "
              f"TrainLoss={avg_train_loss:.6f} | PCC={avg_pcc:.4f} | MSE={avg_mse:.4f}")

        if avg_pcc > best_pcc:
            best_pcc = avg_pcc
            torch.save(
                {"vision": model.state_dict(), "head": head.state_dict()},
                OUTPUT_DIR / "best_vision_probe_routeA.pt",
            )
            print(f"  >> New best PCC={best_pcc:.4f} (checkpoint saved: best_vision_probe_routeA.pt)")

    # ---- Save curves ----
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title("Train Loss (Route A)")
    plt.xlabel("Epoch")

    plt.subplot(1, 3, 2)
    plt.plot(test_pccs)
    plt.title("Test PCC (Route A)")
    plt.xlabel("Epoch")

    plt.subplot(1, 3, 3)
    plt.plot(test_mses)
    plt.title("Test MSE (Route A)")
    plt.xlabel("Epoch")

    curve_path = OUTPUT_DIR / "finetune_curves_routeA.png"
    plt.savefig(curve_path, dpi=150)
    print(f"[DONE] Curves saved to {curve_path}")
    print(f"[DONE] Best PCC (Route A) = {best_pcc:.4f}")


if __name__ == "__main__":
    main()
