import pandas as pd
import numpy as np
import scanpy as sc
import os
from tqdm import tqdm

# === 1. 读取表达数据 ===
expr_path = "/data1/xiaoxinyu/project/gene_text_pairs/gene2go/common_genes_with_expr.csv"
df = pd.read_csv(expr_path, sep=',') 

# 保证 Description 为字符串
df['Description'] = df['Description'].astype(str)

# 设置为 Series 格式
gtex_mean_expr = pd.Series(df['val'].astype(float).values, index=df['Description'].astype(str))

# 所有基因名称（顺序很重要）
symbol_list = gtex_mean_expr.index.tolist()

# === 2. 输出目录 ===
output_dir = "/data1/xiaoxinyu/project/gene_text_pairs/gene2go/gene-h5ad_1_3"
os.makedirs(output_dir, exist_ok=True)

# === 3. 构建并保存每个伪表达谱 ===
for symbol in tqdm(symbol_list, desc="生成伪表达谱"):

    # 构建一个表达值数组（，其他不变）
    expr_values = []
    for gene in symbol_list:
        val = gtex_mean_expr[gene]
        if gene == symbol:
            val *= 10.0  # 目标基因 ×10
        else:
            noise = np.random.normal(0, val * 0.05)  # 加入 5% 高斯噪声
            val = max(val + noise, 0.0)  # 防止出现负表达值
        expr_values.append(val)

    expr_vec = np.array(expr_values).reshape(1, -1)

    # 构建 AnnData
    obs = pd.DataFrame(index=[f'sample_{symbol}'])
    var = pd.DataFrame(index=symbol_list)
    adata = sc.AnnData(X=expr_vec, obs=obs, var=var)

    # 保存
    adata.write_h5ad(f"{output_dir}/{symbol}.h5ad")
