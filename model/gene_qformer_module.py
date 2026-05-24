# /data2/xiaoxinyu/project/model_merged_v75/gene_qformer_module.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    # transformers is optional for "read config only" behavior
    from transformers import BertConfig, BertModel
except Exception:
    BertConfig = None
    BertModel = None


class _QFormerBlock(nn.Module):
    """
    A lightweight Q-Former block:
      - self-attention on queries
      - cross-attention: queries attend to gene tokens (kv)
      - FFN
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_q3 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,               # [B, Nq, D]
        kv: torch.Tensor,                    # [B, L,  D]
        kv_key_padding_mask: Optional[torch.Tensor] = None,  # [B, L], True for PAD
    ) -> torch.Tensor:
        # ---- Query self-attn ----
        q = self.norm_q1(queries)
        q2, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + q2

        # ---- Cross-attn: queries attend to kv ----
        q = self.norm_q2(queries)
        q2, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=kv_key_padding_mask,  # True for PAD
            need_weights=False,
        )
        queries = queries + q2

        # ---- FFN ----
        q = self.norm_q3(queries)
        queries = queries + self.ffn(q)
        return queries


class GeneQFormerBiomedBERT(nn.Module):
    """
    Gene Q-Former bridge module.

    Why the name includes "BiomedBERT":
    - Some papers initialize Q-Former from (BioMed)BERT.
    - In your setup, you said you've merged BERT weights already, so
      you can set load_pretrained_bert=False to avoid remote loading.
    - We still optionally read the BERT config (hidden size, num layers, num heads)
      to keep hyperparams consistent.

    Inputs:
      gene_tokens: [B, L, gene_in_dim] (e.g., 512)
      gene_pad_mask: [B, L] bool, True indicates PAD positions (optional)

    Output:
      q_tokens: [B, num_queries, hidden] (e.g., [B,32,768])
    """

    def __init__(
        self,
        biomedbert_name: str = "",
        gene_in_dim: int = 512,
        hidden: int = 768,
        num_queries: int = 32,
        num_layers: int = 4,
        num_heads: int = 12,
        dropout: float = 0.1,
        load_pretrained_bert: bool = False,
    ):
        super().__init__()

        # Optionally read BERT config to align hyperparams
        if biomedbert_name and BertConfig is not None:
            try:
                cfg = BertConfig.from_pretrained(biomedbert_name)
                # Only override if caller didn't explicitly set hidden/layers/heads
                # (We treat passed args as authoritative; config is a fallback.)
                # Still, it's useful to sanity-check.
                if hidden != cfg.hidden_size:
                    # keep user's hidden, but this can warn in logs if you want
                    pass
                if num_heads != cfg.num_attention_heads:
                    pass
                # if num_layers passed as default 4, but config has 12, you may want 12:
                # We won't override automatically to avoid surprising behavior.
            except Exception:
                cfg = None
        else:
            cfg = None

        self.gene_in_dim = int(gene_in_dim)
        self.hidden = int(hidden)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        # Project gene token dim -> qformer hidden dim
        self.gene_kv_proj = nn.Sequential(
            nn.LayerNorm(self.gene_in_dim),
            nn.Linear(self.gene_in_dim, self.hidden),
        )

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_queries, self.hidden) * 0.02
        )

        # Q-Former blocks
        self.blocks = nn.ModuleList(
            [_QFormerBlock(dim=self.hidden, num_heads=self.num_heads, dropout=dropout)
             for _ in range(self.num_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden)

        # Optional: keep a BERTModel around (NOT used by default)
        # If you later want to initialize weights from BERT, you can implement it here.
        self._bert = None
        if load_pretrained_bert:
            if BertModel is None:
                raise RuntimeError("transformers is not available, cannot load pretrained BERT.")
            self._bert = BertModel.from_pretrained(biomedbert_name)

            # NOTE: We do NOT directly plug BERT forward in this module,
            # because BERT doesn't have cross-attn blocks by default.
            # If you want to copy weights, implement a mapping routine
            # (self-attn weights can be copied block-wise).
            # For now, we just keep it loaded so you can manually inspect/copy.

    def forward(
        self,
        gene_tokens: torch.Tensor,  # [B, L, gene_in_dim]
        gene_pad_mask: Optional[torch.Tensor] = None,  # [B, L] bool, True for PAD
    ) -> torch.Tensor:
        if gene_tokens.dim() != 3:
            raise ValueError(f"gene_tokens must be 3D [B,L,C], got {tuple(gene_tokens.shape)}")
        B, L, C = gene_tokens.shape
        if C != self.gene_in_dim:
            raise ValueError(
                f"gene_tokens last dim={C} != gene_in_dim={self.gene_in_dim}. "
                f"Check Nicheformer output dim."
            )

        if gene_pad_mask is not None:
            if gene_pad_mask.shape != (B, L):
                raise ValueError(
                    f"gene_pad_mask shape {tuple(gene_pad_mask.shape)} != (B,L)=({B},{L})"
                )
            gene_pad_mask = gene_pad_mask.to(dtype=torch.bool, device=gene_tokens.device)

        kv = self.gene_kv_proj(gene_tokens)  # [B, L, hidden]

        queries = self.query_tokens.expand(B, -1, -1).contiguous()  # [B, Nq, hidden]

        for blk in self.blocks:
            queries = blk(queries, kv, kv_key_padding_mask=gene_pad_mask)

        return self.out_norm(queries)  # [B, Nq, hidden]
