# batch_text_embedding_fast.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import sys
sys.path.append("/data2/xiaoxinyu/project")  # dataset.py 所在目录
from dataset import QueryDataset, data_collator_query


# ---------------- 配置 ----------------
input_dir = '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/desc-gpt'
output_dir = '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/text-embeddings'
os.makedirs(output_dir, exist_ok=True)

model_name = "/data1/xiaoxinyu/huggingface/minicpm-v-2_6"
device = "cuda"
batch_size = 64  # 可根据显存调节

# ---------------- 加载模型 ----------------
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ---------------- 函数 ----------------
def get_text_embeddings(texts):
    """批量获取文本嵌入"""
    dataset = QueryDataset(texts, tokenizer, llm_type='qwen2')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: data_collator_query(x)
    )

    all_embeddings = []

    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(data=batch, use_cache=False, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1].mean(dim=1)  # [batch, dim]
                all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)  # [num_texts, dim]

# ---------------- 主流程 ----------------
files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

# 按 batch 处理多个文本文件
batch_texts = []
batch_output_paths = []

for fname in tqdm(files):
    symbol = fname.replace('.txt', '')
    text_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, f"{symbol}_text.pt")

    if os.path.exists(output_path):
        continue  # 已生成跳过

    with open(text_path, 'r') as f:
        desc = f.read()

    batch_texts.append(desc)
    batch_output_paths.append(output_path)

    # 达到 batch_size 或最后一批
    if len(batch_texts) >= batch_size or fname == files[-1]:
        embeddings = get_text_embeddings(batch_texts)
        for emb, out_path in zip(embeddings, batch_output_paths):
            torch.save(emb, out_path)
        batch_texts = []
        batch_output_paths = []
