import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
import os

# 设定路径
# adata_path = "/data2/xiaoxinyu/project/embedding-cosine/DLPFC/top2000/merge_data/merged_stdata_with_label.h5ad"
# output_gene_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-h5ad"
# output_text_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/desc"
adata_path = "/data1/xiaoxinyu/benchmark/DLPFC/stRNA/10xformat/151674/filtered_feature_bc_matrix.h5ad"
output_gene_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/151674/gene-h5ad"
output_text_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/151674/desc"

# 创建目录
os.makedirs(output_gene_dir, exist_ok=True)
os.makedirs(output_text_dir, exist_ok=True)


# 准备数据（每列为一个 spot，每行为一个基因）
adata = sc.read_h5ad(adata_path)

# 若是稀疏矩阵
X_dense = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

# AnnData 默认 X 是 sample × gene
# 但 GSEApy 要求 gene × sample
expr_df = pd.DataFrame(X_dense.T, index=adata.var_names, columns=adata.obs_names)

# ssGSEA 分析
ssgsea_res = gp.ssgsea(
    data=expr_df,
    gene_sets='KEGG_2021_Human', 
    sample_norm_method='rank',
    outdir=None,
    verbose=True
)

# 输出：ssgsea_res.res2d 是 (pathway × spot) 的打分表
# 转换为 scores_df（spot × pathway）
scores_df = ssgsea_res.res2d.pivot(index='Term', columns='Name', values='ES').T
# print(scores_df.shape) 
# (4789, 306) 


def generate_text_descriptions(scores_df, k=10):
    """
    输入:
        scores_df: pd.DataFrame, shape = (n_samples, n_pathways)
        k: int, 选取 top-k 的通路

    输出:
        descriptions: List[str], 每个 sample 对应一条自然语言描述
    """
    descriptions = []

    for idx, row in scores_df.iterrows():
        top_pathways = row.sort_values(ascending=False).head(k).index.tolist()

        desc = [f"The '{path}' pathway is highly enriched." for path in top_pathways]
        full_sentence = " ".join(desc)

        descriptions.append(full_sentence)

    return descriptions

# 生成描述文本
descriptions = generate_text_descriptions(scores_df, k=5)

# 拆分并保存
# 第 i 行表达向量存在 spot_000i.h5ad
# 第 i 条描述文本写入 spot_000i.txt

# for i in range(adata.n_obs):
for i in range(3):  # 先保存前 10 个
    spot_id = adata.obs_names[i][0]  # 直接用 obs_names

    # 保存 gene 表达向量
    adata_single = adata[i].copy()
    adata_single.write_h5ad(os.path.join(output_gene_dir, f"{spot_id}.h5ad"))

    # 保存描述文本
    text_path = os.path.join(output_text_dir, f"{spot_id}.txt")
    with open(text_path, "w") as f:
        f.write(descriptions[i])