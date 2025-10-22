import gzip
import pandas as pd

# 路径
gene2go_path = "/data1/xiaoxinyu/project/gene-text/NCBI/gene2go.gz"
gene_info_path = "/data1/xiaoxinyu/project/gene-text/NCBI/gene_info.gz"

# 读取 gene2go（只保留 GeneID 与 GO 相关列）
gene2go_cols = ['#tax_id', 'GeneID', 'GO_ID', 'Evidence', 'Qualifier', 'GO_term', 'PubMed', 'Category']
gene2go = pd.read_csv(gene2go_path, sep='\t', comment='#', names=gene2go_cols)

# 读取 gene_info（GeneID 和 Symbol）
gene_info_cols = ['#tax_id', 'GeneID', 'Symbol']
gene_info = pd.read_csv(gene_info_path, sep='\t', comment='#', usecols=[0,1,2], names=gene_info_cols)

# 合并 gene2go 和 gene_info，添加 Symbol 列
merged = pd.merge(gene2go, gene_info, on='GeneID', how='left')

# 示例：输出前几行结果
print(merged[['GeneID', 'Symbol', 'GO_ID', 'GO_term']].head())

# 保存结果
merged.to_csv("/data1/xiaoxinyu/project/gene-text/NCBI/gene2go_with_symbol.gz", sep='\t', index=False)
