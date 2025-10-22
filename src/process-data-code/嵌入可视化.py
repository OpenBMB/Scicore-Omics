import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import os

# 1. 加载所有pt文件
folder_path = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/text-embeddings"
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
embeddings = [torch.load(os.path.join(folder_path, f)) for f in pt_files]

# # 2. 计算两两余弦相似度(需要行数相同)
# similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
# for i in range(len(embeddings)):
#     for j in range(len(embeddings)):
#         # 将张量转换为numpy数组并展平
#         vec1 = embeddings[i].detach().numpy().reshape(1, -1)
#         vec2 = embeddings[j].detach().numpy().reshape(1, -1)
#         similarity_matrix[i,j] = cosine_similarity(vec1, vec2)[0][0]
#
# print("余弦相似度矩阵:")
# print(similarity_matrix)

# 3. UMAP降维可视化
# 合并所有嵌入
all_embeddings = torch.cat(embeddings, dim=0).detach().numpy()

# 执行UMAP降维
reducer = umap.UMAP()
# reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)

embedding_2d = reducer.fit_transform(all_embeddings)

# 可视化
plt.figure(figsize=(10, 8))
for i, file in enumerate(pt_files):
    start = i * len(embeddings[0])
    end = start + len(embeddings[0])
    plt.scatter(embedding_2d[start:end, 0], 
                embedding_2d[start:end, 1], 
                # label=file, 
                alpha=0.6)

plt.title('UMAP Visualization of Embeddings')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.savefig(os.path.join(folder_path, '/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/umap_text-embeddings-raw.png'))
plt.show()