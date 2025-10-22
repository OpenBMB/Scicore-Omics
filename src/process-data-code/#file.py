# ========== 查看 pt 文件 =============

import torch

# 加载文件
data = torch.load("/data2/xiaoxinyu/project/gene_text_pairs/gene2go/gene_text_pairs_1.pt")
# data = torch.load("/data1/xiaoxinyu/gene_text_pairs.pt")

# 查看数据结构
print(f"数据类型: {type(data)}")
print(f"数据长度: {len(data)}")

# 查看第一个元素的结构
print("\n第一个元素的结构:")
print(data[0].keys())
# 查看具体内容示例
print("\n示例数据:")
print(f"基因符号: {data[0]['symbol']}")
print(f"基因嵌入维度: {len(data[0]['gene_embedding'])}")
print(f"文本嵌入维度: {len(data[0]['text_embedding'])}")


# =================================

import torch

# 加载基因嵌入文件
gene_embedding = torch.load("/data1/xiaoxinyu/project/gene_text_pairs/gene/HMGCS2_gene.pt")

# 查看嵌入数据
print("基因嵌入数据类型:", type(gene_embedding))
print("基因嵌入数据形状:", gene_embedding.shape)  # 应该是 (512,) 的向量
print("基因嵌入前10个值:", gene_embedding[:10])

# ================ 查看 h5ad文件 =====================

import anndata as ad

# 加载 h5ad 文件
adata = ad.read_h5ad("/data1/xiaoxinyu/project/gene_text_pairs/gene-h5ad/A1BG.h5ad")

# 查看基本信息
print("AnnData 对象:")
print(adata)

# 查看观察值 (obs)
print("\n观察值 (样本信息):")
print(adata.obs)

# 查看变量 (var)
print("\n变量 (基因信息):")
print(adata.var)

# 查看表达矩阵
print("\n表达矩阵 (X):")
print(adata.X)

# 查看特定基因的表达
print("\nA1BG 基因的表达:")
print(adata[:, "A1BG"].X)
_embedding'])}")
    print(f"文本嵌入维度: {len(data[0]['text_embedding'])}")


    
# ============== 查看 json 文件内容 =============

import json

# 加载 JSON 文件
with open("/data1/xiaoxinyu/project/gene-text/NCBI/human_symbol_desc.json", 'r') as f:
    json_data = json.load(f)    

# 查看 JSON 数据结构
print("JSON 数据类型:", type(json_data))
print("JSON 数据长度:", len(json_data))


# ========= gz ============

# 查看有多少条数据
zcat /data1/xiaoxinyu/project/gene-text/NCBI/gene_info.gz | grep -v '^#' | wc -l
# 查看文件前几行
zcat /data1/xiaoxinyu/project/gene-text/NCBI/gene_info.gz | head -n 6

# 查看文件里面tax_id="9606"的数据条有多少
zcat /data1/xiaoxinyu/project/gene-text/NCBI/gene2go/gene2go_with_symbol.gz | awk -F'\t' '$1 == 9606' | wc -l