# batch_gene_embedding_fast.py

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import anndata as ad
import numpy as np
from tqdm import tqdm

# ---------- 配置 ----------
# input_dir = '/data2/xiaoxinyu/project/embedding-cosine/DLPFC/top2000/151674/layer'
# output_dir = '/data2/xiaoxinyu/project/embedding-cosine/DLPFC/embedding/151674/layer-ft'
input_dir = '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-h5ad'
output_dir = '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-embeddings-raw'
os.makedirs(output_dir, exist_ok=True)

backbone_path = "/data2/xiaoxinyu/nicheformer"
# ckpt_path = "/data2/xiaoxinyu/nicheformer/planB_stageA_ckpt/planB_stage1_epoch10.pt"
technology_mean_path = '/home/xiaoxinyu/nicheformer/data/model_means/xenium_mean_script.npy'

# ---------- 模型加载 ----------
print("Loading pretrained model and tokenizer ...")
model = AutoModelForMaskedLM.from_pretrained(
    backbone_path,
    trust_remote_code=True,
    local_files_only=True
).eval().to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    backbone_path,
    trust_remote_code=True,
    local_files_only=True
)

technology_mean = np.load(technology_mean_path)
tokenizer._load_technology_mean(technology_mean)

# ---------- 加载微调权重 ----------
# print(f"Loading finetuned weights from {ckpt_path} ...")
# ckpt = torch.load(ckpt_path, map_location="cuda")

# # 只加载 backbone，过滤掉 head
# backbone_state = {
#     k: v for k, v in ckpt["backbone_state_dict"].items()
#     if not (k.startswith("lm_head") or k.startswith("cls") or "fc" in k)
# }
# missing, unexpected = model.load_state_dict(backbone_state, strict=False)
# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)

# model.eval()


# ---------- 获取单个文件 embeddings（批量化内部实现） ----------
def get_gene_embedding(h5ad_path, batch_size=512):
    adata = ad.read_h5ad(h5ad_path)
    inputs = tokenizer(adata)  # 假设返回 dict: input_ids, attention_mask

    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    for batch_input_ids, batch_attention_mask in dataloader:
        batch_input_ids = batch_input_ids.to("cuda")
        batch_attention_mask = batch_attention_mask.to("cuda")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                embeddings = model.get_embeddings(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    layer=-1,
                    with_context=False
                )   # 预期 [batch, seq_len, 512]

                if embeddings.dim() == 3:
                    embeddings = embeddings.mean(dim=1)  # → [batch, 512]
                elif embeddings.dim() == 2:
                    # 保险起见，保证维度是 [batch, 512]
                    pass
                else:
                    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

                all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"✅ {os.path.basename(h5ad_path)} embedding shape: {all_embeddings.shape}")
    return all_embeddings


# ---------- 批量处理目录 ----------
files = [f for f in os.listdir(input_dir) if f.endswith('.h5ad')]
for fname in tqdm(files):
    symbol = fname.replace('.h5ad', '')
    inp = os.path.join(input_dir, fname)
    out = os.path.join(output_dir, f"{symbol}_gene.pt")

    if os.path.exists(out):
        print(f"⚠️ 已存在，跳过: {out}")
        continue

    embedding = get_gene_embedding(inp, batch_size=1024)
    torch.save(embedding, out)
    print(f"💾 Embedding saved: {out}, shape: {embedding.shape}")
