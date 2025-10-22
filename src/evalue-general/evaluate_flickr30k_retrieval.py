# ======================================================
# evaluate_flickr30k_retrieval.py
# 任务：使用 MiniCPM-V 提取图像/文本特征并计算 Recall@K
# ======================================================

import torch
from torch.utils.data import DataLoader
from flickr30k_dataset import Flickr30kDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import json

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    image_ids = [b["image_id"] for b in batch]
    captions = [b["captions"] for b in batch]
    return {"image": images, "image_id": image_ids, "captions": captions}


# ========== 配置部分 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/data2/xiaoxinyu/project/model" # 替换路径
image_dir = "/data2/xiaoxinyu/data/flickr30k_images/flickr30k_images/flickr30k_images"
jsonl_path = "/data2/xiaoxinyu/data/flickr30k_images/ln_flickr30k_val_captions_multi.jsonl"

# ========== 1. 加载模型 ==========
print("🚀 加载模型中...")
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("✅ 模型加载完成！")

# ========== 2. 加载数据 ==========
dataset = Flickr30kDataset(image_dir=image_dir, jsonl_path=jsonl_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
print(f"✅ 数据集加载完成，共 {len(dataset)} 张图片")

# ========== 3. 提取图像特征 ==========
image_feats, image_ids = [], []
for batch in tqdm(dataloader, desc="🖼️ 提取图像特征"):
    images = batch["image"].to(device)
    with torch.no_grad():
        feats = model.get_vllm_embedding(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    image_feats.append(feats.cpu())
    image_ids.extend(batch["image_id"])
image_feats = torch.cat(image_feats, dim=0)

# ========== 4. 提取文本特征 ==========
refs = {}
with open(jsonl_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        refs.setdefault(obj["image_id"], []).append(obj["caption"].strip())

text_feats, text_to_img = [], []
for img_id, captions in tqdm(refs.items(), desc="💬 提取文本特征"):
    for cap in captions:
        text_to_img.append(img_id)
        with torch.no_grad():
            feat = model.encode_text(tokenizer(cap, return_tensors="pt").to(device))
            feat = feat / feat.norm(dim=-1, keepdim=True)
        text_feats.append(feat.cpu())
text_feats = torch.cat(text_feats, dim=0)

np.savez("flickr30k_minicpm_features.npz",
         image_feats=image_feats.numpy(),
         text_feats=text_feats.numpy(),
         image_ids=np.array(image_ids),
         text_to_img=np.array(text_to_img))
print("✅ 特征已保存到 flickr30k_minicpm_features.npz")

# ========== 5. 计算 Recall@K ==========
def l2norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
I, T = l2norm(image_feats.numpy()), l2norm(text_feats.numpy())
sim = I @ T.T

def image_to_text_metrics(sim, img_ids, text_to_img, ks=(1,5,10)):
    ranks = []
    for i in range(sim.shape[0]):
        sims = sim[i]
        sorted_idx = np.argsort(-sims)
        correct = [j for j, tid in enumerate(text_to_img) if tid == img_ids[i]]
        rank_pos = [np.where(sorted_idx == c)[0][0] for c in correct]
        ranks.append(min(rank_pos))
    ranks = np.array(ranks)
    result = {f"R@{k}": np.mean(ranks < k)*100 for k in ks}
    result["MedR"] = np.median(ranks)+1
    return result

metrics = image_to_text_metrics(sim, image_ids, text_to_img)
print("\n🎯 图文检索结果：")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")
