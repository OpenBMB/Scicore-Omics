# model_utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE
import seaborn as sns
import umap

import swanlab

# ============ Dataset =============
class GeneTextDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = torch.tensor(self.data[idx]['gene_embedding'], dtype=torch.float32)
        t = torch.tensor(self.data[idx]['text_embedding'], dtype=torch.float32)
        g = g.squeeze(0) 
        t = t.squeeze(0) 
        return g, t

class GeneTextDatasetWithlabel(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path)

        # 统计所有字符串标签
        all_labels = [d['region_label'] for d in self.data]
        unique_labels = sorted(set(all_labels))

        # 建立 label -> id 的映射
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = torch.tensor(self.data[idx]['gene_embedding'], dtype=torch.float32)
        t = torch.tensor(self.data[idx]['text_embedding'], dtype=torch.float32)

        if torch.isnan(t).any() or torch.isinf(t).any():
            # 方案1: 返回全零，避免 NaN
            t = torch.zeros_like(t)
            # 方案2: 随机生成一个 embedding
            # t = torch.randn_like(t)
        
        # region_label 先转 id
        raw_label = self.data[idx]['region_label']
        label_id = self.label2id[raw_label]
        label = torch.tensor(label_id, dtype=torch.long)

        return g, t, label


# # ============ Mapping MLP ============
# class MappingNet(nn.Module):
#     def __init__(self, in_dim=512, out_dim=3584, hidden_dims=[512, 1024, 2048]): 
#         super().__init__()
#         layers = []
#         dims = [in_dim] + hidden_dims + [out_dim]
#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i+1]))
#             if i < len(dims) - 2:
#                 layers.append(nn.ReLU())
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)


# ============ Mapping MLP (改进版) ============
class MappingNet(nn.Module):
    def __init__(self, in_dim=512, out_dim=3584, hidden_dims=[1024, 2048]):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))       # 防止过拟合
                layers.append(nn.LayerNorm(dims[i+1]))  # 稳定训练

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)  # 投影后归一化
    

# ============ VSE++ Loss ============
def vse_plus_plus_loss(emb1, emb2, margin=0.2):
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    sim = torch.matmul(emb1, emb2.T)
    batch_size = emb1.size(0)

    pos_sim = sim.diag()
    term1 = margin + sim - pos_sim.unsqueeze(1)
    term1 = F.relu(term1) * (1 - torch.eye(batch_size, device=emb1.device))
    loss1 = term1.sum(dim=1).mean()

    term2 = margin + sim.T - pos_sim.unsqueeze(1)
    term2 = F.relu(term2) * (1 - torch.eye(batch_size, device=emb1.device))
    loss2 = term2.sum(dim=1).mean()

    return (loss1 + loss2) / 2


# ============ InfoNCE Loss ============
def info_nce_loss(emb1, emb2, temperature=0.07):
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)

    logits = torch.matmul(emb1, emb2.T) / temperature
    labels = torch.arange(emb1.size(0), device=emb1.device)
    loss_i2t = F.cross_entropy(logits, labels)     # gene -> text
    loss_t2i = F.cross_entropy(logits.T, labels)   # text -> gene
    return (loss_i2t + loss_t2i) / 2


# ============ Cosine Loss ============
def cosine_loss(emb1, emb2):
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    cos_sim = torch.sum(emb1 * emb2, dim=1)  # [batch_size]
    loss = 1 - cos_sim.mean()                # 余弦越大损失越小
    return loss


# ============ MSE Loss ============
def mse_loss(emb1, emb2):
    return F.mse_loss(emb1, emb2)


# ============ Training ============
def train(model, dataloader, epochs=30, lr=1e-4, loss_type="all"):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0

        for gene_embed, text_embed in dataloader:
            gene_embed, text_embed = gene_embed.cuda(), text_embed.cuda()
            pred = model(gene_embed)

            # --- loss 选择 ---
            if loss_type == "vse":
                loss = vse_plus_plus_loss(pred, text_embed)
            elif loss_type == "nce":
                loss = info_nce_loss(pred, text_embed)
            elif loss_type == "cosine":
                loss = cosine_loss(pred, text_embed)
            elif loss_type == "mse":
                loss = mse_loss(pred, text_embed)
            else:  # 混合
                loss = (
                    1.0 * info_nce_loss(pred, text_embed)
                    + 0.1 * vse_plus_plus_loss(pred, text_embed)
                    + 0.5 * cosine_loss(pred, text_embed)
                    + 0.5 * mse_loss(pred, text_embed)
                )

            # --- 优化 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

        scheduler.step()
        avg_loss = total_loss / len(dataloader)

        print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f}")

        swanlab.log({
            "loss": avg_loss,
            "epoch": epoch + 1
        })


# ============ Evaluate ============
def evaluate(model, dataloader, method="pca", save_path=None):
    model.eval()
    all_gene, all_text = [], []

    with torch.no_grad():
        for g, t in dataloader:
            all_gene.append(model(g.cuda()))
            all_text.append(t.cuda())

    all_gene = torch.cat(all_gene)
    all_text = torch.cat(all_text)

    all_gene = F.normalize(all_gene, p=2, dim=1)
    all_text = F.normalize(all_text, p=2, dim=1)

    sim = torch.matmul(all_gene, all_text.T)  # [N, N]
    ranks = torch.argsort(sim.cpu(), dim=1, descending=True)
    targets = torch.arange(sim.size(0), device=sim.device)

    # Recall@1, @5, @10
    recalls = {}
    for k in [1, 5, 10]:
        correct_at_k = (ranks[:, :k] == targets.cpu().unsqueeze(1)).any(dim=1).float().mean().item()
        # correct_at_k = (ranks[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        recalls[f"Recall@{k}"] = correct_at_k
        print(f"🎯 Recall@{k}: {correct_at_k:.4f}")
        swanlab.log({f"recall@{k}": correct_at_k})

    # MRR
    def compute_mrr(ranks, targets):
        rr = 1.0 / (ranks == targets.cpu().unsqueeze(1)).nonzero(as_tuple=False)[:,1].float().add(1.0)
        return rr.mean().item()

    mrr = compute_mrr(ranks, targets)
    print(f"🔁 MRR: {mrr:.4f}")
    swanlab.log({"mrr": mrr})

    # mAP
    sim_np = sim.detach().cpu().numpy()
    y_true = np.eye(sim_np.shape[0])
    map_score = np.mean([average_precision_score(y_true[i], sim_np[i]) for i in range(len(y_true))])
    print(f"📊 mAP: {map_score:.4f}")
    swanlab.log({"mAP": map_score})

    # cosine similarity
    cosine_sim = (all_gene * all_text).sum(dim=1)  
    avg_cos_sim = cosine_sim.mean().item()
    print(f"🧲 Cos_Sim: {avg_cos_sim:.4f}")
    swanlab.log({"avg_cosine_similarity": avg_cos_sim})

    # PCA / tsne
    visualize_embeddings(all_gene, all_text, method=method, save_path=save_path)

# ============ PCA 可视化 ============
def visualize_embeddings(gene_emb, text_emb, method="pca", save_path=None):
    emb = torch.cat([gene_emb, text_emb], dim=0).cpu().numpy()
    labels = ["gene"] * gene_emb.size(0) + ["text"] * text_emb.size(0)

    # 降维
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(emb)

    # 可视化
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"gene": "blue", "text": "red"}

    for label in ["gene", "text"]:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(reduced[idx, 0], reduced[idx, 1], c=colors[label], label=label, alpha=0.6)

    ax.legend()
    ax.set_title(f"{method.upper()} of Gene/Text Embeddings")
    if save_path:
        plt.savefig(save_path)
    swanlab.log({f"{method}_plot": swanlab.Image(fig)})
    plt.close()