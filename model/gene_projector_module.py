# gene_projector_module.py
import torch.nn as nn

class GeneProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=3584, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        # x: [B, 32, 768]  ->  [B, 32, 3584]
        return self.proj(x)
