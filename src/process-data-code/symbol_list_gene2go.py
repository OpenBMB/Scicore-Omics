# save_gene_symbol_desc.py
import gzip
import json
import os

gene_info_path = "/data1/xiaoxinyu/project/gene-text/NCBI/gene2go/gene2go_with_symbol.gz"
output_path = "/data1/xiaoxinyu/project/gene-text/NCBI/gene2go/human_symbol_desc.json"

# 使用集合来存储唯一的元组 (symbol, desc)
unique_pairs = set()

with gzip.open(gene_info_path, 'rt') as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split('\t')
        if parts[0] == "9606":
            symbol = parts[9]
            desc = parts[5]  
            # 检查并去除重复项
            unique_pairs.add((symbol, desc))

# 转换为列表以便保存为 JSON
symbol_desc_list = list(unique_pairs)

# 保存为 JSON 文件
with open(output_path, "w") as f:
    json.dump(symbol_desc_list, f, indent=2)

print(f"✅ Saved {len(symbol_desc_list)} gene symbol-description pairs to {output_path}")

# >>> print("JSON 数据长度:", len(json_data))