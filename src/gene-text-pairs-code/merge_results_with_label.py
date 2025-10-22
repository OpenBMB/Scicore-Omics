# merge_results_with_label.py
import os
import torch
from tqdm import tqdm
import scanpy as sc

# 路径
adata_path = "/data2/xiaoxinyu/project/embedding-cosine/DLPFC/top2000/merge_data/merged_stdata_with_label.h5ad"
gene_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-embeddings"
text_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/text-embeddings"
output_path = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene_text_pairs_with_label.pt"

# 读取 AnnData
adata = sc.read_h5ad(adata_path)

# 提取 label 映射
# obs_names 应该和你保存时的 spot_00000 对应
label_dict = {spot_id: adata.obs.loc[spot_id, "region_label"] for spot_id in adata.obs_names}

# 获取所有 gene embedding 文件对应的 spot_id
symbols = [f.replace('_gene.pt', '') for f in os.listdir(gene_dir) if f.endswith('_gene.pt')]

# 控制最多 150 个
# symbols = symbols[:300]

results = []
for symbol in tqdm(symbols):
    text_path = os.path.join(text_dir, f"{symbol}_text.pt")
    if not os.path.exists(text_path):
        continue

    gene_embedding = torch.load(os.path.join(gene_dir, f"{symbol}_gene.pt"))
    text_embedding = torch.load(text_path)

    # 取出对应 label（symbol = spot_xxxxx）
    region_label = label_dict.get(symbol, None)

    results.append({
        "symbol": symbol,
        "gene_embedding": gene_embedding.tolist(),
        "text_embedding": text_embedding.tolist(),
        "region_label": region_label
    })

torch.save(results, output_path)
print(f"✅ 汇总完成，共 {len(results)} 对，已包含 region_label")
