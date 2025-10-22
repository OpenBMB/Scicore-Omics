# merge_results.py
import os
import torch
from tqdm import tqdm

gene_dir = '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-embeddings-epoch20'
text_dir = '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/text-embeddings'
symbols = [f.replace('_gene.pt', '') for f in os.listdir(gene_dir) if f.endswith('_gene.pt')]


results = []
# max_pairs = 2000
# count = 0  # 新增：用于计数实际处理的对数

for symbol in tqdm(symbols):
    text_path = os.path.join(text_dir, f"{symbol}_text.pt")
    if not os.path.exists(text_path):  # 新增：检查文本嵌入文件是否存在
        continue
        
    gene_embedding = torch.load(os.path.join(gene_dir, f"{symbol}_gene.pt"))
    text_embedding = torch.load(text_path)
    results.append({
        "symbol": symbol,
        "gene_embedding": gene_embedding.tolist(),
        "text_embedding": text_embedding.tolist()
    })
    
    # count += 1
    # if count >= max_pairs:  # 修改：使用实际处理对数来判断
    #     break



torch.save(results, '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-with-complex-text/gene_text_pairs_ft20_gpt_all.pt')
print(f"✅ 汇总完成，共 {len(results)} 对")
