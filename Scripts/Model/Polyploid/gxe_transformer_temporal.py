"""
GxE TRANSFORMER WITH TEMPORAL ENVIRONMENTAL ENCODING
====================================================

This is the IMPROVED model architecture for beating RÃƒâ€šÃ‚Â²=0.52 baseline.

Key improvements:
1. Temporal LSTM encoder for time-series environmental data (20 weeks ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 6 features)
2. Location and year embeddings (categorical variables)
3. Population structure embedding (from admixture analysis)
4. Explicit GxE interaction modeling (bilinear)
5. Cross-attention between genomic and environmental modalities

Expected RÃƒâ€šÃ‚Â²: 0.52-0.66 (beating EcoPopDL-GP's 0.52)

Author: Modified for rice flowering time prediction
Date: 2025
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Recommended regularization by component (used by training loop, not enforced here)
E_ONLY_WEIGHT_DECAY = 1.0
G_ONLY_WEIGHT_DECAY = 1.0
GXE_WEIGHT_DECAY = 0.01
E_ONLY_DROPOUT = 0.0
G_ONLY_DROPOUT = 0.0
GXE_DROPOUT = 0.4
POP_EMBED_WEIGHT_DECAY = 5.0
HOTSPOT_FOCUS_BIAS = 6.0  # bias attention toward hotspot tokens when no_pool_te_hotspots is enabled
RESIDUAL_GATE_INIT = 0.01  # default gating strength for the GxE residual path


# ============================================================================
# GLOBAL ENVIRONMENTAL AXIS CONFIGURATION
# ============================================================================

DEFAULT_CRITICAL_ENV_FEATURES = [
    'daylength_h',
    'tmax_C',
    'tmin_C',
    'gdd',
    'vpd_kPa',
    'srad_allsky'
]

AXIS_FEATURE_GROUPS: Dict[str, List[str]] = {
    "daylength": ["daylength"],
    "temperature": ["tmax", "tmin", "gdd", "srad"],
    "water": ["vpd", "precip", "rain"]
}


def token_dropout(
    genomic: torch.Tensor,
    pad_mask: torch.Tensor,
    p_drop: float = 0.10,
    keep_first_token: bool = True,
    min_keep_tokens: Optional[int] = None,
    return_drop_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Token-dropout augmentation for hierarchical tensors [B,C,T,F].
    pad_mask: True = padding.
    """
    if p_drop <= 0.0:
        return genomic, pad_mask, None if return_drop_mask else None
    B, C, T, _ = genomic.shape
    if min_keep_tokens is None:
        min_keep_tokens = max(64, int(0.01 * C * T))
    valid = ~pad_mask
    drop = (torch.rand(valid.shape, device=genomic.device) < p_drop) & valid
    if keep_first_token and T > 0:
        drop[..., 0] = False
    pad_aug = pad_mask | drop
    kept = (~pad_aug).sum(dim=(1, 2))
    too_sparse = kept < min_keep_tokens
    if too_sparse.any():
        drop[too_sparse] = False
        pad_aug = pad_mask | drop
    genomic_aug = genomic.masked_fill(drop.unsqueeze(-1), 0.0)
    return genomic_aug, pad_aug, (drop if return_drop_mask else None)


# ============================================================================
# ENVIRONMENT-AWARE NORMALIZATION + ADVERSARY
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    Uses sinusoidal encoding based on SNP positions.
    """

    def __init__(self, d_model, dropout=0.1, max_len=120000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = int(d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)
        # Seed buffer to a reasonable length; extend dynamically if needed.
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            # Dynamically extend on the fly to avoid crashes for long sequences.
            position = torch.arange(seq_len, device=x.device).unsqueeze(1)
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * self.div_term)
            pe[:, 1::2] = torch.cos(position * self.div_term)
        else:
            pe = self.pe[:seq_len, :].to(x.device)
        x = x + pe.unsqueeze(0)
        return self.dropout(x)


# ============================================================================
# ATTENTION POOLING
# ============================================================================

class AttentionPooling(nn.Module):
    """
    Single-head attention pooling. Mask expects True on padded positions.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.att = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=None, bias=None):
        # x: [B,L,D], mask: [B,L] with True for padding
        scores = self.att(x).squeeze(-1)  # [B,L]
        if bias is not None:
            scores = scores + bias
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B,L,1]
        return (x * weights).sum(dim=1)  # [B,D]

class FeatureGroupEmbedder(nn.Module):
    """
    Project genomic features with separate heads per feature group.
    Optionally apply log1p scaling to distance-like channels.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        feature_groups: Optional[Dict[str, List[int]]] = None,
        log1p_indices: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self._log1p_indices = sorted({int(i) for i in (log1p_indices or []) if 0 <= int(i) < self.input_dim})
        self._group_indices: Dict[str, torch.Tensor] = {}
        self.group_layers = nn.ModuleDict()
        if feature_groups:
            for name, idxs in feature_groups.items():
                if not idxs:
                    continue
                clean = sorted({int(i) for i in idxs if 0 <= int(i) < self.input_dim})
                if not clean:
                    continue
                self._group_indices[name] = torch.tensor(clean, dtype=torch.long)
                self.group_layers[name] = nn.Sequential(
                    nn.Linear(len(clean), embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        self.out_norm = nn.LayerNorm(embed_dim) if self.group_layers else None
        self.fallback = None
        if not self.group_layers:
            self.fallback = nn.Sequential(
                nn.Linear(self.input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._log1p_indices:
            x = x.clone()
            idx = torch.tensor(self._log1p_indices, device=x.device, dtype=torch.long)
            vals = torch.index_select(x, dim=-1, index=idx)
            vals = torch.log1p(vals.clamp(min=0.0))
            x.index_copy_(-1, idx, vals)
        if self.fallback is not None:
            return self.fallback(x)
        out = x.new_zeros(x.size(0), x.size(1), self.embed_dim)
        for name, layer in self.group_layers.items():
            idx = self._group_indices[name].to(x.device)
            out = out + layer(torch.index_select(x, dim=-1, index=idx))
        return self.out_norm(out) if self.out_norm is not None else out

class BiologicalAwareEmbedding(nn.Module):
    """
    Biological-aware embedding for dosage, distance context, and hotspot indicators.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        if embed_dim < 4:
            raise ValueError("embed_dim must be >= 4 for BiologicalAwareEmbedding.")
        self.embed_dim = int(embed_dim)
        dosage_dim = max(1, self.embed_dim // 2)
        context_dim = max(1, self.embed_dim // 4)
        hotspot_dim = max(1, self.embed_dim - dosage_dim - context_dim)
        self.dosage_head = nn.Embedding(3, dosage_dim)
        self.context_head = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim)
        )
        self.hotspot_head = nn.Linear(2, hotspot_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )

    def forward(
        self,
        dosage: torch.Tensor,
        distances: torch.Tensor,
        hotspots: torch.Tensor
    ) -> torch.Tensor:
        dosage_emb = self.dosage_head(dosage.long())
        context_emb = self.context_head(distances.float())
        hotspot_emb = self.hotspot_head(hotspots.float())
        combined = torch.cat([dosage_emb, context_emb, hotspot_emb], dim=-1)
        return self.fusion(combined)

class LowRankBilinear(nn.Module):
    """
    Factorized bilinear layer: x^T W y with W = U V^T (rank-r) per output dim.
    """
    def __init__(self, in1: int, in2: int, out: int, rank: int = 8, bias: bool = True):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LowRankBilinear.")
        self.in1 = in1
        self.in2 = in2
        self.out = out
        self.rank = int(rank)
        self.left = nn.Linear(in1, out * self.rank, bias=False)
        self.right = nn.Linear(in2, out * self.rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(out)) if bias else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        left = self.left(x).view(bsz, self.out, self.rank)
        right = self.right(y).view(bsz, self.out, self.rank)
        out = (left * right).sum(dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def regularization(self) -> torch.Tensor:
        return 0.5 * (self.left.weight.abs().mean() + self.right.weight.abs().mean())

class DenseBlockAggregator(nn.Module):
    """
    Aggregate SNP embeddings into dense haplotype block tokens using scatter_add.
    Block IDs may be sparse; they are compacted per batch to avoid large empty ranges.
    """
    def __init__(self, feature_dim: int, block_embed_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, block_embed_dim) if feature_dim != block_embed_dim else nn.Identity()

    def forward(
        self,
        genomic_features: torch.Tensor,
        block_membership_ids: torch.Tensor,
        genomic_mask: torch.Tensor,
        exclude_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            genomic_features: [B, L, F]
            block_membership_ids: [B, L] int, -1 for invalid/padded
            genomic_mask: [B, L] bool, True for padding
        Returns:
            block_embeddings: [B, N_blocks, block_embed_dim]
            block_padding_mask: [B, N_blocks] True for empty blocks
        """
        B, L, F = genomic_features.shape
        device = genomic_features.device
        dtype = genomic_features.dtype

        if genomic_mask is None:
            genomic_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        if exclude_mask is None:
            exclude_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        block_ids = block_membership_ids.long()
        valid = (~genomic_mask) & (block_ids >= 0) & (~exclude_mask)

        if not valid.any():
            block_embeddings = genomic_features.new_zeros((B, 1, F))
            block_padding_mask = torch.ones((B, 1), device=device, dtype=torch.bool)
            return self.proj(block_embeddings), block_padding_mask

        flat_features = genomic_features[valid]               # [N_valid, F]
        flat_block_ids = block_ids[valid]                     # [N_valid]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)[valid]

        # Compact per-batch block IDs to avoid huge sparse ranges causing OOM.
        pairs = torch.stack([batch_idx, flat_block_ids], dim=1)  # [N_valid, 2]
        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        max_block_id = flat_block_ids.max().long()
        sort_keys = unique_pairs[:, 0].long() * (max_block_id + 1) + unique_pairs[:, 1].long()
        sorted_idx = torch.argsort(sort_keys)
        sorted_pairs = unique_pairs[sorted_idx]
        batch_sorted = sorted_pairs[:, 0]
        counts = torch.bincount(batch_sorted, minlength=B)
        offsets = torch.cumsum(counts, dim=0) - counts
        local_rank_sorted = torch.arange(sorted_pairs.size(0), device=device) - offsets[batch_sorted]
        local_rank = torch.empty_like(local_rank_sorted)
        local_rank[sorted_idx] = local_rank_sorted
        local_block_ids = local_rank[inverse]
        max_blocks = int(counts.max().item()) if counts.numel() else 1
        if max_blocks <= 0:
            max_blocks = 1

        flat_index = batch_idx * max_blocks + local_block_ids  # [N_valid]

        block_sums = torch.zeros((B * max_blocks, F), device=device, dtype=dtype)
        block_counts = torch.zeros((B * max_blocks, 1), device=device, dtype=dtype)

        block_sums.scatter_add_(0, flat_index.unsqueeze(-1).expand(-1, F), flat_features)
        ones = torch.ones_like(flat_block_ids, device=device, dtype=dtype).unsqueeze(-1)
        block_counts.scatter_add_(0, flat_index.unsqueeze(-1), ones)

        block_embeddings = block_sums / block_counts.clamp(min=1)
        block_embeddings = block_embeddings.view(B, max_blocks, F)
        block_padding_mask = block_counts.view(B, max_blocks).squeeze(-1) == 0
        return self.proj(block_embeddings), block_padding_mask


class SparseSNPInjector(nn.Module):
    """
    Extract individual hotspot SNP embeddings as sparse tokens.
    """
    def __init__(self, feature_dim: int, snp_embed_dim: int, max_sparse_tokens: Optional[int] = None):
        super().__init__()
        self.proj = nn.Linear(feature_dim, snp_embed_dim) if feature_dim != snp_embed_dim else nn.Identity()
        self.max_sparse_tokens = max_sparse_tokens

    def forward(
        self,
        genomic_features: torch.Tensor,
        is_hotspot: torch.Tensor,
        genomic_mask: torch.Tensor,
        func_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            genomic_features: [B, L, F]
            is_hotspot: [B, L] bool
            genomic_mask: [B, L] bool, True for padding
            func_type_ids: [B, L] long optional functional type ids
        Returns:
            sparse_snp_tokens: [B, Ls, snp_embed_dim]
            sparse_snp_mask: [B, Ls] True for padding
            sparse_snp_is_hotspot: [B, Ls] bool
            sparse_snp_func_type: [B, Ls] long (or None)
        """
        B, L, F = genomic_features.shape
        device = genomic_features.device

        if genomic_mask is None:
            genomic_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        sparse_select = is_hotspot & (~genomic_mask)
        counts = sparse_select.sum(dim=1) if sparse_select.numel() else torch.zeros((B,), device=device)
        max_sparse = int(counts.max().item()) if counts.numel() else 0
        if self.max_sparse_tokens is not None:
            max_sparse = min(max_sparse, int(self.max_sparse_tokens))
        if max_sparse <= 0:
            max_sparse = 1

        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).expand(B, L)

        # if func_type_ids is not None:
        #     # Priority: promoter (2) > genic (1) > TE (3) > others
        #     ft = func_type_ids.float()
        #     priority = torch.zeros_like(ft)
        #     priority = torch.where(ft == 2, torch.full_like(priority, 3.0), priority)
        #     priority = torch.where(ft == 1, torch.full_like(priority, 2.0), priority)
        #     priority = torch.where(ft == 3, torch.full_like(priority, 1.0), priority)
        #     scores = priority - 1e-6 * pos  # tie-break by position (earlier slightly favored)
        # else:
        scores = -pos

        scores = torch.where(sparse_select, scores, pos.new_full((B, L), -float("inf")))
        topk = torch.topk(scores, k=max_sparse, dim=1)
        idx = topk.indices  # [B, max_sparse]

        sparse_tokens = genomic_features.gather(1, idx.unsqueeze(-1).expand(-1, -1, F))
        sparse_is_hotspot = is_hotspot.gather(1, idx)

        valid = torch.isfinite(topk.values)
        sparse_tokens = sparse_tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        sparse_is_hotspot = sparse_is_hotspot & valid
        sparse_mask = ~valid

        sparse_func_type = None
        if func_type_ids is not None:
            sparse_func_type = func_type_ids.gather(1, idx)
            sparse_func_type = sparse_func_type.masked_fill(~valid, 0)

        sparse_tokens = self.proj(sparse_tokens)
        return sparse_tokens, sparse_mask, sparse_is_hotspot, sparse_func_type


class HotspotAwareBlockEmbedding(nn.Module):
    """
    Combine dense haplotype block embeddings with sparse hotspot SNP tokens.
    """
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        max_sparse_tokens: Optional[int] = None,
        num_func_types: int = 4,
        use_func_type_embed: bool = True
    ):
        super().__init__()
        self.dense_aggregator = DenseBlockAggregator(feature_dim=feature_dim, block_embed_dim=embed_dim)
        self.sparse_injector = SparseSNPInjector(
            feature_dim=feature_dim,
            snp_embed_dim=embed_dim,
            max_sparse_tokens=max_sparse_tokens
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.token_type_embed = nn.Embedding(3, embed_dim)
        self.num_func_types = int(num_func_types)
        self.func_type_embed = nn.Embedding(self.num_func_types, embed_dim) if use_func_type_embed else None

    def _aggregate_block_func_types(
        self,
        func_type_ids: torch.Tensor,
        block_membership_ids: torch.Tensor,
        genomic_mask: Optional[torch.Tensor],
        block_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, L = block_membership_ids.shape
        device = block_membership_ids.device
        if genomic_mask is None:
            genomic_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        block_ids = block_membership_ids.long()
        valid = (~genomic_mask) & (block_ids >= 0)

        if not valid.any():
            return torch.zeros((B, 1), device=device, dtype=torch.long)

        flat_block_ids = block_ids[valid]
        flat_func = func_type_ids[valid].clamp(min=0, max=self.num_func_types - 1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)[valid]

        pairs = torch.stack([batch_idx, flat_block_ids], dim=1)  # [N_valid, 2]
        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        max_block_id = flat_block_ids.max().long()
        sort_keys = unique_pairs[:, 0].long() * (max_block_id + 1) + unique_pairs[:, 1].long()
        sorted_idx = torch.argsort(sort_keys)
        sorted_pairs = unique_pairs[sorted_idx]
        batch_sorted = sorted_pairs[:, 0]
        counts = torch.bincount(batch_sorted, minlength=B)
        offsets = torch.cumsum(counts, dim=0) - counts
        local_rank_sorted = torch.arange(sorted_pairs.size(0), device=device) - offsets[batch_sorted]
        local_rank = torch.empty_like(local_rank_sorted)
        local_rank[sorted_idx] = local_rank_sorted
        local_block_ids = local_rank[inverse]
        max_blocks = int(counts.max().item()) if counts.numel() else 1
        if max_blocks <= 0:
            max_blocks = 1

        flat_index = batch_idx * max_blocks + local_block_ids  # [N_valid]
        one_hot = F.one_hot(flat_func, num_classes=self.num_func_types).float()
        block_presence = torch.zeros((B * max_blocks, self.num_func_types), device=device, dtype=one_hot.dtype)
        block_presence.scatter_add_(0, flat_index.unsqueeze(-1).expand(-1, self.num_func_types), one_hot)
        present = block_presence > 0
        priorities = torch.arange(self.num_func_types, device=device, dtype=block_presence.dtype)
        block_type = (present * priorities).max(dim=-1).values.long()
        block_type = block_type.view(B, max_blocks)
        if block_padding_mask is not None:
            expected = block_padding_mask.size(1)
            if block_type.size(1) < expected:
                pad = expected - block_type.size(1)
                block_type = torch.cat(
                    [block_type, torch.zeros((B, pad), device=device, dtype=torch.long)],
                    dim=1
                )
            elif block_type.size(1) > expected:
                block_type = block_type[:, :expected]
            block_type = block_type.masked_fill(block_padding_mask, 0)
        return block_type

    def forward(
        self,
        genomic_features: torch.Tensor,
        block_membership_ids: torch.Tensor,
        is_hotspot: torch.Tensor,
        genomic_mask: torch.Tensor,
        func_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = genomic_features.size(0)
        device = genomic_features.device

        block_embeddings, block_padding_mask = self.dense_aggregator(
            genomic_features, block_membership_ids, genomic_mask, exclude_mask=is_hotspot
        )
        sparse_tokens, sparse_mask, sparse_is_hotspot, sparse_func_type = self.sparse_injector(
            genomic_features, is_hotspot, genomic_mask, func_type_ids=func_type_ids
        )

        cls_tok = self.cls_token.expand(B, -1, -1)
        cls_mask = torch.zeros((B, 1), device=device, dtype=torch.bool)
        cls_hot = torch.zeros((B, 1), device=device, dtype=torch.bool)

        unified_tokens = torch.cat([cls_tok, block_embeddings, sparse_tokens], dim=1)
        # Token types: 0=cls, 1=block, 2=sparse
        block_len = block_embeddings.size(1)
        sparse_len = sparse_tokens.size(1)
        type_ids = torch.cat(
            [
                torch.zeros((B, 1), device=device, dtype=torch.long),
                torch.ones((B, block_len), device=device, dtype=torch.long),
                torch.full((B, sparse_len), 2, device=device, dtype=torch.long),
            ],
            dim=1
        )
        unified_tokens = unified_tokens + self.token_type_embed(type_ids)
        unified_mask = torch.cat([cls_mask, block_padding_mask, sparse_mask], dim=1)
        unified_hotspot_mask = torch.cat(
            [cls_hot, torch.zeros_like(block_padding_mask, dtype=torch.bool), sparse_is_hotspot],
            dim=1
        )

        func_type_tokens = None
        if func_type_ids is not None:
            block_func_type = self._aggregate_block_func_types(
                func_type_ids, block_membership_ids, genomic_mask, block_padding_mask
            )
            if sparse_func_type is None:
                sparse_func_type = torch.zeros((B, sparse_len), device=device, dtype=torch.long)
            cls_func = torch.zeros((B, 1), device=device, dtype=torch.long)
            func_type_tokens = torch.cat([cls_func, block_func_type, sparse_func_type], dim=1)
            if self.func_type_embed is not None:
                unified_tokens = unified_tokens + self.func_type_embed(func_type_tokens)

        return unified_tokens, unified_mask, unified_hotspot_mask, func_type_tokens


# ============================================================================
# TEMPORAL ENVIRONMENTAL ENCODER
# ============================================================================

class TemporalEnvironmentalEncoder(nn.Module):
    """
    Encode time-series environmental data using Bidirectional LSTM.
    
    Input: [B, n_steps, n_features_per_step] (weekly)
    Output: [B, env_embed_dim]
    
    This captures temporal dynamics of weather during rice growth:
    - Early months: vegetative growth
    - Middle months: floral induction (critical for flowering time!)
    - Late months: reproductive development
    """
    
    def __init__(
        self,
        n_features_per_month=6,  # daylength, tmax, tmin, gdd, vpd, srad
        n_months=20,
        hidden_dim=64,
        num_layers=2,
        env_embed_dim=32,
        dropout=0.3
    ):
        super().__init__()
        
        self.n_features_per_month = n_features_per_month
        self.n_months = n_months
        self.hidden_dim = hidden_dim
        self.env_embed_dim = env_embed_dim
        
        # Bidirectional LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=n_features_per_month,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        # Attention pooling over time
        self.attention_pool = AttentionPooling(hidden_dim * 2)
        # Project to embedding dimension (after pooling)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, env_embed_dim),  # *2 for bidirectional
            nn.LayerNorm(env_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.1)
        )
        self.seq_dim = hidden_dim * 2
        self.global_dim = hidden_dim * 2
        self._last_sequence = None
        self._last_hidden = None

    def forward(self, env_timeseries):
        """
        Args:
            env_timeseries: [B, n_months, n_features_per_month]
        
        Returns:
            env_repr: [B, env_embed_dim]
        """
        batch_size = env_timeseries.size(0)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(env_timeseries)
        # lstm_out: [B, n_months, hidden_dim*2]
        self._last_sequence = lstm_out
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_final = torch.cat([h_forward, h_backward], dim=-1)
        env_repr = self.attention_pool(lstm_out)
        env_repr = self.proj(env_repr)
        self._last_hidden = h_final
        return env_repr

    @property
    def last_sequence(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_sequence", None)

    @property
    def last_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_hidden", None)


class TemporalConvEncoder(nn.Module):
    """
    Temporal convolutional encoder for environmental sequences.
    """
    def __init__(
        self,
        n_features_per_month: int = 6,
        n_months: int = 20,
        conv_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        env_embed_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_features_per_month = n_features_per_month
        self.n_months = n_months
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.env_embed_dim = env_embed_dim
        self.blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = n_features_per_month if i == 0 else conv_channels
            block = nn.Sequential(
                nn.Conv1d(in_ch, conv_channels, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.blocks.append(block)
            if in_ch != conv_channels:
                self.skip_convs.append(nn.Conv1d(in_ch, conv_channels, kernel_size=1))
            else:
                self.skip_convs.append(nn.Identity())
        self.proj = nn.Sequential(
            nn.Linear(conv_channels, env_embed_dim),
            nn.LayerNorm(env_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.seq_dim = conv_channels
        self.global_dim = conv_channels
        self._last_sequence = None
        self._last_hidden = None

    def forward(self, env_timeseries):
        x = env_timeseries.transpose(1, 2)  # [B, features, months]
        for block, skip_conv in zip(self.blocks, self.skip_convs):
            residual = x
            out = block(x)
            residual = skip_conv(residual)
            # Guard against length drift when kernel_size is even (Conv1d padding is asymmetric).
            if out.size(2) != residual.size(2):
                min_t = min(out.size(2), residual.size(2))
                out = out[:, :, :min_t]
                residual = residual[:, :, :min_t]
            x = out + residual
        self._last_sequence = x.transpose(1, 2)
        pooled = x.mean(dim=2)
        self._last_hidden = pooled
        env_repr = self.proj(pooled)
        return env_repr

    @property
    def last_sequence(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_sequence", None)

    @property
    def last_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_hidden", None)


class TemporalPyramidEncoder(nn.Module):
    """
    Multi-scale temporal pyramid encoder to capture short shocks and long trends.
    """
    def __init__(
        self,
        n_features_per_month: int = 6,
        n_months: int = 20,
        conv_channels: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        env_embed_dim: int = 32,
        dropout: float = 0.2,
        scales: Optional[List[int]] = None
    ):
        super().__init__()
        self.n_features_per_month = n_features_per_month
        self.n_months = n_months
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.env_embed_dim = env_embed_dim
        self.scales = [1, 2, 4] if scales is None else [int(s) for s in scales]
        if 1 not in self.scales:
            self.scales = [1] + self.scales

        self.stem = nn.Sequential(
            nn.Conv1d(n_features_per_month, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.blocks = nn.ModuleList()
        for _ in self.scales:
            layers = []
            for _ in range(max(1, num_layers)):
                layers.extend([
                    nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
            self.blocks.append(nn.Sequential(*layers))

        self.proj = nn.Sequential(
            nn.Linear(conv_channels * len(self.scales), env_embed_dim),
            nn.LayerNorm(env_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.seq_dim = conv_channels
        self.global_dim = conv_channels * len(self.scales)
        self._last_sequence = None
        self._last_hidden = None

    @staticmethod
    def _avg_pool(x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale <= 1:
            return x
        length = x.size(-1)
        pad = (scale - (length % scale)) % scale
        if pad:
            x = F.pad(x, (0, pad))
        return F.avg_pool1d(x, kernel_size=scale, stride=scale)

    def forward(self, env_timeseries):
        x = env_timeseries.transpose(1, 2)  # [B, features, months]
        base = self.stem(x)
        seq = None
        pooled = []
        for scale, block in zip(self.scales, self.blocks):
            x_s = self._avg_pool(base, scale)
            x_s = block(x_s)
            pooled.append(x_s.mean(dim=2))
            if scale == 1:
                seq = x_s
            else:
                up = F.interpolate(x_s, size=base.size(2), mode="linear", align_corners=False)
                seq = seq + up if seq is not None else up
        if seq is None:
            seq = base
        self._last_sequence = seq.transpose(1, 2)
        global_feat = torch.cat(pooled, dim=-1)
        self._last_hidden = global_feat
        env_repr = self.proj(global_feat)
        return env_repr

    @property
    def last_sequence(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_sequence", None)

    @property
    def last_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_hidden", None)


class WideEnvMLPEncoder(nn.Module):
    """
    Simple MLP encoder for wide environment vectors (no temporal structure).
    """
    def __init__(
        self,
        n_features_per_month: int,
        env_embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_months = 1
        self.seq_dim = 1
        self.global_dim = env_embed_dim
        self._last_sequence = None
        self._last_hidden = None
        self.net = nn.Sequential(
            nn.Linear(n_features_per_month, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_embed_dim),
            nn.LayerNorm(env_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

    def forward(self, env_vec: torch.Tensor) -> torch.Tensor:
        if env_vec.dim() == 3:
            env_vec = env_vec.squeeze(1)
        out = self.net(env_vec)
        self._last_sequence = None
        self._last_hidden = out
        return out

    @property
    def last_sequence(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_sequence", None)

    @property
    def last_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_hidden", None)


class FlatEnvMLPEncoder(nn.Module):
    """
    MLP encoder over flattened temporal environment tensors.
    """
    def __init__(
        self,
        n_months: int,
        n_features_per_month: int,
        env_embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        n_layers: int = 2
    ):
        super().__init__()
        self.n_months = int(n_months)
        self.seq_dim = 1
        self.global_dim = env_embed_dim
        self._last_sequence = None
        self._last_hidden = None
        in_dim = self.n_months * int(n_features_per_month)
        n_layers = max(2, int(n_layers))
        layers: List[nn.Module] = []
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        layers.extend([
            nn.Linear(hidden_dim, env_embed_dim),
            nn.LayerNorm(env_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, env_ts: torch.Tensor) -> torch.Tensor:
        if env_ts.dim() == 3:
            env_ts = env_ts.reshape(env_ts.size(0), -1)
        out = self.net(env_ts)
        self._last_sequence = None
        self._last_hidden = out
        return out

    @property
    def last_sequence(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_sequence", None)

    @property
    def last_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_hidden", None)


# ============================================================================
class ChromoAwareTransformer(nn.Module):
    """
    Chromosome-aware encoder: processes each chromosome independently with a Transformer,
    then aggregates chromosome embeddings. Designed for tensor inputs only.
    """

    def __init__(
        self,
        feature_dim: int,
        feature_groups: Optional[Dict[str, List[int]]] = None,
        feature_log1p_indices: Optional[List[int]] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        downsample_stride: int = 1,
        downsample_kernel: Optional[int] = None,
        block_id_channel_idx: Optional[int] = None,
        num_chromosomes: Optional[int] = None,
        env_seq_dim: Optional[int] = None,
        use_env_cross_attention: bool = False,
        env_embed_dim: Optional[int] = None,
        use_env_film: bool = False,
        use_env_pool_bias: bool = False,
        meta_embed_dim: Optional[int] = None,
        use_meta_film: bool = False,
        meta_film_scale: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_feature_dim = int(feature_dim)
        self.use_pos_encoding = use_pos_encoding
        self.downsample_stride = max(1, int(downsample_stride))
        k = int(downsample_kernel) if downsample_kernel is not None else self.downsample_stride
        self.downsample_kernel = max(1, k)
        self.block_id_channel_idx = block_id_channel_idx if block_id_channel_idx is not None else None
        self.num_chromosomes = num_chromosomes
        self.chrom_weight_logits = None
        if num_chromosomes is not None:
            # Learnable per-chromosome mixing weights to avoid diluting strong signals via a flat mean.
            init = torch.zeros(num_chromosomes, dtype=torch.float)
            self.chrom_weight_logits = nn.Parameter(init)
        # Optional env cross-attention (per-SNP interaction before pooling).
        self.use_env_cross_attention = bool(use_env_cross_attention)
        self.env_seq_proj = None
        self.env_cross_attention = None
        self.env_cross_norm = None
        if self.use_env_cross_attention and env_seq_dim is not None:
            self.env_seq_proj = nn.Linear(env_seq_dim, embed_dim)
            self.env_cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.env_cross_norm = nn.LayerNorm(embed_dim)
        self.use_env_film = bool(use_env_film and env_embed_dim is not None)
        self.env_to_gamma = None
        self.env_to_beta = None
        if self.use_env_film:
            self.env_to_gamma = nn.Sequential(
                nn.Linear(env_embed_dim, embed_dim),
                nn.Tanh()
            )
            self.env_to_beta = nn.Sequential(
                nn.Linear(env_embed_dim, embed_dim),
                nn.Tanh()
            )
        self.use_env_pool_bias = bool(use_env_pool_bias and env_embed_dim is not None)
        self.env_to_pool_bias = None
        if self.use_env_pool_bias:
            self.env_to_pool_bias = nn.Sequential(
                nn.Linear(env_embed_dim, embed_dim),
                nn.Tanh()
            )
        self.use_meta_film = bool(use_meta_film and meta_embed_dim is not None)
        self.meta_film_scale = float(meta_film_scale)
        self.meta_to_gamma = None
        self.meta_to_beta = None
        if self.use_meta_film:
            self.meta_to_gamma = nn.Sequential(
                nn.Linear(meta_embed_dim, embed_dim),
                nn.Tanh()
            )
            self.meta_to_beta = nn.Sequential(
                nn.Linear(meta_embed_dim, embed_dim),
                nn.Tanh()
            )
        self.input_proj = FeatureGroupEmbedder(
            input_dim=self.input_feature_dim,
            embed_dim=embed_dim,
            feature_groups=feature_groups,
            log1p_indices=feature_log1p_indices,
            dropout=dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout) if use_pos_encoding else None
        self.pool = AttentionPooling(embed_dim)
        self.downsample = None
        if self.downsample_stride > 1:
            self.downsample = nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=self.downsample_kernel,
                stride=self.downsample_stride,
                padding=0,
                groups=embed_dim,
                bias=True
            )

    def _pool_by_block(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
        hotspot_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pools consecutive SNP tokens that share the same block_id_raw (>0).
        Non-block tokens (<=0) are kept as individual tokens. Returns pooled seq and mask.
        """
        device = seq.device
        B, T, F = seq.shape
        block_idx = self.block_id_channel_idx
        pooled_list = []
        mask_list = []
        lengths = []
        for b in range(B):
            valid = ~mask[b] if mask is not None else torch.ones(T, dtype=torch.bool, device=device)
            tokens = seq[b][valid]
            hot = hotspot_mask[b][valid] if hotspot_mask is not None else None
            if tokens.numel() == 0:
                pooled = seq.new_zeros((0, F))
            else:
                bids = tokens[:, block_idx].round().long()
                pooled_rows: list[torch.Tensor] = []
                i = 0
                while i < tokens.size(0):
                    bid = bids[i]
                    j = i + 1
                    while j < tokens.size(0) and bids[j] == bid:
                        j += 1
                    chunk = tokens[i:j]
                    if bid > 0:
                        if hot is not None:
                            hot_chunk = hot[i:j]
                            if hot_chunk.any():
                                if (~hot_chunk).any():
                                    pooled_rows.append(chunk[~hot_chunk].mean(dim=0, keepdim=False))
                                pooled_rows.extend(chunk[hot_chunk])
                            else:
                                pooled_rows.append(chunk.mean(dim=0, keepdim=False))
                        else:
                            pooled_rows.append(chunk.mean(dim=0, keepdim=False))
                    else:
                        pooled_rows.extend(chunk)  # keep non-block tokens as-is
                    i = j
                pooled = torch.stack(pooled_rows) if pooled_rows else seq.new_zeros((0, F))
            pooled_list.append(pooled)
            mask_list.append(torch.zeros(pooled.size(0), dtype=torch.bool, device=device))
            lengths.append(pooled.size(0))
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            max_len = 1
        out_seq = seq.new_zeros((B, max_len, F))
        out_mask = torch.ones((B, max_len), dtype=torch.bool, device=device)
        for b, pooled in enumerate(pooled_list):
            L = pooled.size(0)
            if L == 0:
                continue
            out_seq[b, :L] = pooled
            out_mask[b, :L] = mask_list[b]
        return out_seq, out_mask

    @staticmethod
    def _normalize_pad_mask(mask: Optional[torch.Tensor], genomic: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert saved masks (1=real, 0=pad) or bool masks (True=pad) into bool pad masks.
        """
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            pad_mask = mask
        else:
            pad_mask = (mask == 0)
        return pad_mask.to(device=genomic.device, dtype=torch.bool)

    def forward(
        self,
        genomic: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        env_seq: Optional[torch.Tensor] = None,
        env_mod: Optional[torch.Tensor] = None,
        mod_scale: Optional[torch.Tensor] = None,
        env_repr: Optional[torch.Tensor] = None,
        meta_repr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            genomic: [B, C, T, F]
            mask:    [B, C, T] True for padding
            env_seq: [B, L_env, D_env] optional environmental tokens for per-SNP cross-attn
            env_mod: [B, embed_dim] optional environment modulation vector
            mod_scale: scalar tensor to scale env_mod influence
        Returns:
            global_repr: [B, D] aggregated across chromosomes
            per_chrom_repr: [B, C, D] per-chromosome embeddings
        """
        if genomic.dim() != 4:
            raise RuntimeError(f"Expected genomic tensor [B,C,T,F], got {tuple(genomic.shape)}")
        mask = self._normalize_pad_mask(mask, genomic)
        genomic = torch.nan_to_num(genomic, nan=0.0)
        if mask is not None:
            genomic = genomic.masked_fill(mask.unsqueeze(-1), 0.0)
        B, C, T, _ = genomic.shape
        gamma = None
        beta = None
        if self.use_env_film and env_repr is not None and env_repr.dim() == 2:
            gamma = self.env_to_gamma(env_repr).unsqueeze(1)
            beta = self.env_to_beta(env_repr).unsqueeze(1)
        meta_gamma = None
        meta_beta = None
        if self.use_meta_film and meta_repr is not None and meta_repr.dim() == 2:
            meta_gamma = self.meta_to_gamma(meta_repr).unsqueeze(1) * self.meta_film_scale
            meta_beta = self.meta_to_beta(meta_repr).unsqueeze(1) * self.meta_film_scale
        pool_bias_proj = None
        if self.use_env_pool_bias and env_repr is not None and env_repr.dim() == 2:
            pool_bias_proj = self.env_to_pool_bias(env_repr).unsqueeze(1)
        bias_scale = float(self.embed_dim) ** 0.5
        per_chrom = []
        for c in range(C):
            seq = genomic[:, c, :, :]
            pad = mask[:, c, :] if mask is not None else None
            # Optional block-aware pooling using block_id_raw channel (reduces tokens while respecting blocks)
            if self.block_id_channel_idx is not None and self.block_id_channel_idx < seq.size(-1):
                seq, pad = self._pool_by_block(seq, pad)
            expected_dim = self.input_proj.input_dim
            if seq.size(-1) != expected_dim:
                if seq.size(-1) > expected_dim:
                    seq = seq[..., :expected_dim]
                else:
                    seq = F.pad(seq, (0, expected_dim - seq.size(-1)))
            x = self.input_proj(seq)
            if gamma is not None:
                x = x * (1.0 + gamma) + beta
            if meta_gamma is not None:
                x = x * (1.0 + meta_gamma) + meta_beta
            if self.pos_encoding is not None:
                x = self.pos_encoding(x)
            # Optional SNP-level env cross-attention before pooling.
            if (
                self.use_env_cross_attention
                and env_seq is not None
                and self.env_seq_proj is not None
                and self.env_cross_attention is not None
            ):
                env_proj = self.env_seq_proj(env_seq)
                x_env, _ = self.env_cross_attention(query=x, key=env_proj, value=env_proj)
                x = self.env_cross_norm(x + x_env)
            # Lightweight element-wise modulation of SNP tokens by environment embedding.
            if env_mod is not None and mod_scale is not None:
                x = x * (1.0 + mod_scale * env_mod.unsqueeze(1))
            # Optional depthwise strided downsampling (order-preserving) to reduce sequence length before attention.
            if self.downsample is not None and x.size(1) >= self.downsample_kernel:
                x_t = x.transpose(1, 2)  # [B, F, T]
                x_ds = self.downsample(x_t)  # [B, F, T']
                x = x_ds.transpose(1, 2)
                if pad is not None:
                    k = self.downsample_kernel
                    s = self.downsample_stride
                    L = pad.size(1)
                    new_mask = []
                    for start in range(0, L - k + 1, s):
                        win = pad[:, start:start + k]
                        new_mask.append(win.all(dim=1))
                    pad = torch.stack(new_mask, dim=1) if new_mask else pad[:, :0]
            x = self.encoder(x, src_key_padding_mask=pad)
            # Apply environment modulation after encoding so pooled SNPs carry env-conditioned signal.
            if env_mod is not None and mod_scale is not None:
                x = x * (1.0 + mod_scale * env_mod.unsqueeze(1))
            pool_bias = None
            if pool_bias_proj is not None:
                pool_bias = (x * pool_bias_proj).sum(dim=-1) / bias_scale
            per_chrom.append(self.pool(x, mask=pad, bias=pool_bias))
        per_chrom_repr = torch.stack(per_chrom, dim=1)  # [B, C, D]
        if mask is not None:
            chrom_valid = ~mask.view(B, C, T).all(dim=2)
        else:
            chrom_valid = torch.ones((B, C), device=per_chrom_repr.device, dtype=torch.bool)

        if self.chrom_weight_logits is not None:
            logits = self.chrom_weight_logits
            if logits.numel() != C:
                if logits.numel() > C:
                    logits = logits[:C]
                else:
                    pad_len = C - logits.numel()
                    logits = torch.cat([logits, logits.new_zeros(pad_len)], dim=0)
            weights = torch.softmax(logits, dim=0)  # [C]
            weights = weights.view(1, C, 1)
            effective_weights = weights * chrom_valid.unsqueeze(-1).float()
            denom = effective_weights.sum(dim=1).clamp(min=1e-8)
            global_repr = (per_chrom_repr * effective_weights).sum(dim=1) / denom
        else:
            weights = chrom_valid.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1e-8)
            global_repr = (per_chrom_repr * weights).sum(dim=1) / denom
        return global_repr, per_chrom_repr


class GxE_FusionHead(nn.Module):
    """
    Shared fusion + interaction head used by both tensor and raw-genotype models.
    """
    def __init__(
        self,
        embed_dim: int,
        env_embed_dim: int,
        location_embed_dim: int,
        year_embed_dim: int,
        pop_embed_dim: int,
        interaction_dim: int = 64,
        dropout: float = GXE_DROPOUT,
        main_head_dropout: Optional[float] = None,
        interaction_head_dropout: Optional[float] = None,
        low_rank_bilinear_rank: int = 0,
        interaction_reg_lambda: float = 0.0,
        use_gxe_moe: bool = False,
        gxe_moe_num_experts: int = 4,
        gxe_moe_hidden_dim: Optional[int] = None,
        gxe_moe_temperature: float = 1.0,
    ):
        super().__init__()
        self.interaction_dim = int(interaction_dim)
        self.interaction_reg_lambda = float(interaction_reg_lambda)
        self.low_rank_bilinear_rank = int(low_rank_bilinear_rank)
        self.use_gxe_moe = bool(use_gxe_moe)
        self.gxe_moe_temperature = float(gxe_moe_temperature)

        fused_dim = embed_dim + env_embed_dim + location_embed_dim + year_embed_dim + pop_embed_dim
        head_drop = dropout if main_head_dropout is None else main_head_dropout
        self.fuse_dropout = nn.Dropout(head_drop)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(fused_dim, 1)
        )

        if self.low_rank_bilinear_rank > 0:
            self.interaction_bilinear = LowRankBilinear(
                embed_dim, env_embed_dim, self.interaction_dim, rank=self.low_rank_bilinear_rank
            )
        else:
            self.interaction_bilinear = nn.Bilinear(embed_dim, env_embed_dim, self.interaction_dim)
        self.g_proj = nn.Linear(embed_dim, self.interaction_dim)
        self.e_proj = nn.Linear(env_embed_dim, self.interaction_dim)
        inter_hidden = max(64, self.interaction_dim if gxe_moe_hidden_dim is None else gxe_moe_hidden_dim)
        self.interaction_head = nn.Sequential(
            nn.Linear(self.interaction_dim + location_embed_dim + year_embed_dim + pop_embed_dim, inter_hidden),
            nn.LayerNorm(inter_hidden),
            nn.GELU(),
            nn.Dropout(dropout if interaction_head_dropout is None else interaction_head_dropout),
            nn.Linear(inter_hidden, 1)
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(RESIDUAL_GATE_INIT)))

        if self.use_gxe_moe:
            self.gxe_gate = nn.Sequential(
                nn.Linear(pop_embed_dim, max(4, pop_embed_dim)),
                nn.GELU(),
                nn.Linear(max(4, pop_embed_dim), gxe_moe_num_experts)
            )
            self.gxe_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.interaction_dim + location_embed_dim + year_embed_dim + pop_embed_dim, inter_hidden),
                        nn.LayerNorm(inter_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout if interaction_head_dropout is None else interaction_head_dropout),
                        nn.Linear(inter_hidden, 1)
                    )
                    for _ in range(gxe_moe_num_experts)
                ]
            )
        else:
            self.gxe_gate = None
            self.gxe_experts = None

    def interaction_reg_penalty(self) -> Optional[torch.Tensor]:
        if self.interaction_reg_lambda <= 0.0:
            return None
        if hasattr(self.interaction_bilinear, "regularization"):
            return self.interaction_reg_lambda * self.interaction_bilinear.regularization()
        weight = getattr(self.interaction_bilinear, "weight", None)
        if weight is None:
            return None
        return self.interaction_reg_lambda * weight.abs().mean()

    def compute_ge_feat(self, g_repr: torch.Tensor, env_repr: torch.Tensor) -> torch.Tensor:
        return self.interaction_bilinear(g_repr, env_repr) + (self.g_proj(g_repr) * self.e_proj(env_repr))

    def forward(
        self,
        g_repr: torch.Tensor,
        env_repr_main: torch.Tensor,
        loc_emb: torch.Tensor,
        year_emb: torch.Tensor,
        pop_emb: torch.Tensor,
        env_repr_gxe: Optional[torch.Tensor] = None,
        stage: int = 0,
        return_components: bool = False
    ):
        env_int = env_repr_gxe if env_repr_gxe is not None else env_repr_main
        fused = torch.cat([g_repr, env_repr_main, loc_emb, year_emb, pop_emb], dim=-1)
        fused = self.fuse_dropout(fused)
        main_out = self.head(fused).squeeze(-1)

        ge_feat = self.compute_ge_feat(g_repr, env_int)
        inter_input = torch.cat([ge_feat, loc_emb, year_emb, pop_emb], dim=-1)
        if self.use_gxe_moe and self.gxe_gate is not None and self.gxe_experts is not None:
            gate_logits = self.gxe_gate(pop_emb)
            if self.gxe_moe_temperature != 1.0:
                gate_logits = gate_logits / max(1e-6, self.gxe_moe_temperature)
            gate = torch.softmax(gate_logits, dim=-1)
            expert_outs = torch.stack(
                [expert(inter_input).squeeze(-1) for expert in self.gxe_experts],
                dim=-1
            )
            gxe_out = (expert_outs * gate).sum(dim=-1)
        else:
            gxe_out = self.interaction_head(inter_input).squeeze(-1)

        gate = torch.clamp(self.residual_gate, min=0.0)
        gxe_out_scaled = gxe_out * gate

        if stage == 1:
            out = main_out
        elif stage == 2:
            out = gxe_out_scaled
        else:
            out = main_out + gxe_out_scaled

        aux: Dict[str, torch.Tensor] = {}
        penalty = self.interaction_reg_penalty()
        if penalty is not None:
            aux["interaction_reg"] = penalty
        aux["main_out"] = main_out
        aux["gxe_out"] = gxe_out_scaled
        aux["gxe_out_raw"] = gxe_out
        aux["residual_gate"] = gate.detach()

        if return_components:
            return out, aux
        if aux:
            return out, aux
        return out


class GxE_Transformer_Tensor(nn.Module):
    """
    GxE model that consumes ChromoMap hierarchical tensors with HABE.
    Pipeline:
      1) SNP feature projection -> HABE (block + sparse tokens) -> genomic transformer -> pooled genomic repr
      2) TemporalEnvironmentalEncoder -> env embedding
      3) Metadata embeddings (location/year/pop)
      4) Concatenate and predict (+ interaction head)
    """
    def __init__(
        self,
        genomic_feature_dim: int,
        num_chromosomes: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_intra_layers: int = 2,
        num_cross_layers: int = 1,
        ff_dim: int = 512,
        n_env_features_per_month: int = 6,
        n_months: int = 20,
        env_hidden_dim: int = 64,
        env_lstm_layers: int = 2,
        env_embed_dim: int = 32,
        n_locations: int = 3,
        n_years: int = 2,
        location_embed_dim: int = 8,
        year_embed_dim: int = 4,
        n_populations: int = 6,
        pop_embed_dim: int = 16,
        dropout: float = GXE_DROPOUT,
        main_head_dropout: Optional[float] = None,
        interaction_head_dropout: Optional[float] = None,
        residual_gate_init: float = 0.01,
        distance_log1p: bool = True,
        use_env_anomalies: bool = False,
        env_anomaly_mean: Optional[np.ndarray] = None,
        add_row_embeddings: bool = True,
        row_embed_dim: int = 32,
        cross_chrom_attention: bool = False,
        chr_downsample_stride: int = 1,
        chr_downsample_kernel: Optional[int] = None,
        block_id_channel_idx: Optional[int] = None,
        te_hotspot_channel_idx: Optional[int] = None,
        te_channel_idx: Optional[int] = None,
        genic_channel_idx: Optional[int] = None,
        promoter_channel_idx: Optional[int] = None,
        dosage_channel_idx: Optional[int] = None,
        te_distance_channel_idx: Optional[int] = None,
        gene_distance_channel_idx: Optional[int] = None,
        dosage_genic_channel_idx: Optional[int] = None,
        dosage_promoter_channel_idx: Optional[int] = None,
        dosage_te_channel_idx: Optional[int] = None,
        block_gene_count_channel_idx: Optional[int] = None,
        block_snp_density_channel_idx: Optional[int] = None,
        block_mean_maf_channel_idx: Optional[int] = None,
        use_habe: bool = True,
        use_biological_aware_embedding: bool = True,
        max_sparse_tokens: Optional[int] = None,
        hotspot_focus_bias: float = HOTSPOT_FOCUS_BIAS,
        use_functional_hotspots: bool = True,
        use_functional_token_types: bool = True,
        use_functional_pool_bias: bool = True,
        func_hotspot_quantile: float = 0.75,
        func_const_thresh: float = 0.98,
        func_bias_scale: float = 0.2,
        modality_dropout_p: float = 0.1,
        interaction_dim: int = 64,
        use_env_cross_attention: bool = False,
        use_env_film: bool = True,
        use_env_pool_bias: bool = True,
        use_meta_film: bool = True,
        meta_film_scale: float = 0.1,
        interaction_reg_lambda: float = 0.0,
        low_rank_bilinear_rank: Optional[int] = None,
        use_gxe_moe: bool = False,
        gxe_moe_num_experts: int = 4,
        gxe_moe_hidden_dim: Optional[int] = None,
        gxe_moe_temperature: float = 1.0,
        env_encoder_type: str = "lstm",
        env_conv_channels: int = 64,
        env_conv_layers: int = 3,
        env_conv_kernel: int = 3,
        env_pyramid_scales: Optional[List[int]] = None,
        env_pyramid_layers: Optional[int] = None,
        use_dosage_branch: bool = False,
        dosage_branch_hidden: int = 200,
        dosage_gate_hidden: int = 32,
        dosage_gate_dropout: float = 0.1,
        dosage_blend_prior: float = 0.8,
        dosage_fixed_weight: Optional[float] = None,
        dosage_train_weight: Optional[float] = None,
        dosage_eval_weight: Optional[float] = None,
        dosage_append_realcount: bool = False,
        dosage_center: bool = True,
        dosage_scale: bool = True,
        dosage_pca_components: Optional[torch.Tensor] = None,
        dosage_pca_mean: Optional[torch.Tensor] = None,
        dosage_pca_std: Optional[torch.Tensor] = None,
        n_aux_targets: int = 0
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.env_embed_dim = int(env_embed_dim)
        self.location_embed_dim = int(location_embed_dim)
        self.year_embed_dim = int(year_embed_dim)
        self.pop_embed_dim = int(pop_embed_dim)
        self.interaction_dim = int(interaction_dim)
        self.use_habe = bool(use_habe)
        self.uses_habe = self.use_habe
        self.block_id_channel_idx = block_id_channel_idx if block_id_channel_idx is not None else None
        self.te_hotspot_channel_idx = te_hotspot_channel_idx if te_hotspot_channel_idx is not None else None
        self.te_channel_idx = te_channel_idx if te_channel_idx is not None else None
        self.genic_channel_idx = genic_channel_idx if genic_channel_idx is not None else None
        self.promoter_channel_idx = promoter_channel_idx if promoter_channel_idx is not None else None
        self.dosage_channel_idx = dosage_channel_idx if dosage_channel_idx is not None else None
        self.te_distance_channel_idx = te_distance_channel_idx if te_distance_channel_idx is not None else None
        self.gene_distance_channel_idx = gene_distance_channel_idx if gene_distance_channel_idx is not None else None
        self.dosage_genic_channel_idx = (
            dosage_genic_channel_idx if dosage_genic_channel_idx is not None else None
        )
        self.dosage_promoter_channel_idx = (
            dosage_promoter_channel_idx if dosage_promoter_channel_idx is not None else None
        )
        self.dosage_te_channel_idx = (
            dosage_te_channel_idx if dosage_te_channel_idx is not None else None
        )
        self.block_gene_count_channel_idx = (
            block_gene_count_channel_idx if block_gene_count_channel_idx is not None else None
        )
        self.block_snp_density_channel_idx = (
            block_snp_density_channel_idx if block_snp_density_channel_idx is not None else None
        )
        self.block_mean_maf_channel_idx = (
            block_mean_maf_channel_idx if block_mean_maf_channel_idx is not None else None
        )
        self.max_sparse_tokens = max_sparse_tokens
        # Hard cap for stability on small datasets
        if self.max_sparse_tokens is None or self.max_sparse_tokens > 1024:
            self.max_sparse_tokens = 1024
        self.strict_hotspots = True  # enable strict hotspot selection for small/simple traits
        self.n_months = int(n_months)
        self.hotspot_focus_bias = float(hotspot_focus_bias)
        self.hotspot_focus_bias_param = nn.Parameter(torch.tensor(float(hotspot_focus_bias)), requires_grad=True)
        self.use_functional_hotspots = bool(use_functional_hotspots)
        self.use_functional_token_types = bool(use_functional_token_types)
        self.use_functional_pool_bias = bool(use_functional_pool_bias)
        self.func_hotspot_quantile = float(func_hotspot_quantile)
        self.func_const_thresh = float(func_const_thresh)
        self.func_bias_scale = float(func_bias_scale)
        self._warned_no_hotspot_source = False
        self._warned_non_integer_blocks = False
        self.num_func_types = 4
        self.register_buffer(
            "func_type_bias_weights",
            torch.tensor([0.0, 0.5, 0.5, 0.3], dtype=torch.float32)
        )
        self.modality_dropout_p = modality_dropout_p
        self.interaction_dim = interaction_dim
        self.use_env_film = bool(use_env_film)
        self.use_env_pool_bias = bool(use_env_pool_bias)
        self.use_meta_film = bool(use_meta_film)
        self.meta_film_scale = float(meta_film_scale)
        self.interaction_reg_lambda = float(interaction_reg_lambda)
        self.low_rank_bilinear_rank = int(low_rank_bilinear_rank) if low_rank_bilinear_rank else 0
        self.use_gxe_moe = bool(use_gxe_moe)
        self.gxe_moe_num_experts = max(1, int(gxe_moe_num_experts))
        self.gxe_moe_temperature = float(gxe_moe_temperature) if gxe_moe_temperature else 1.0
        self.distance_log1p = bool(distance_log1p)
        self.env_to_gamma = None
        self.env_to_beta = None
        self.env_to_pool_bias = None
        self.meta_to_gamma = None
        self.meta_to_beta = None
        self.use_bae = bool(use_biological_aware_embedding) and self.use_habe
        self.genomic_bae = None
        if self.use_bae and embed_dim < 4:
            logging.warning("BAE disabled: embed_dim < 4.")
            self.use_bae = False
        if self.use_bae:
            if (
                self.dosage_channel_idx is None
                or self.te_distance_channel_idx is None
                or self.gene_distance_channel_idx is None
            ):
                logging.warning(
                    "BAE enabled but missing dosage/te_dist/gene_dist channel indices; disabling BAE."
                )
                self.use_bae = False
            else:
                self.genomic_bae = BiologicalAwareEmbedding(embed_dim=embed_dim)
        self.use_dosage_branch = bool(use_dosage_branch) and self.dosage_channel_idx is not None
        self.dosage_branch_hidden = int(dosage_branch_hidden)
        self.dosage_gate_hidden = max(2, int(dosage_gate_hidden))
        self.dosage_gate_dropout = float(dosage_gate_dropout)
        self.dosage_blend_prior = float(dosage_blend_prior)
        self.dosage_fixed_weight = None if dosage_fixed_weight is None else float(dosage_fixed_weight)
        self.dosage_fixed_weight_train = float(dosage_train_weight) if dosage_train_weight is not None else None
        self.dosage_fixed_weight_eval = float(dosage_eval_weight) if dosage_eval_weight is not None else None
        if (
            self.dosage_fixed_weight_train is None
            and self.dosage_fixed_weight_eval is None
            and self.dosage_fixed_weight is not None
        ):
            self.dosage_fixed_weight_train = self.dosage_fixed_weight
            self.dosage_fixed_weight_eval = self.dosage_fixed_weight
        self.dosage_append_realcount = bool(dosage_append_realcount)
        self.dosage_center = bool(dosage_center)
        self.dosage_scale = bool(dosage_scale)
        self.dosage_context_dim = (
            self.env_embed_dim + self.location_embed_dim + self.year_embed_dim + self.pop_embed_dim
        )
        self.simple_branch: Optional[nn.Sequential] = None
        self.dosage_context_branch: Optional[nn.Sequential] = None
        self.dosage_gate: Optional[nn.Sequential] = None
        self.dosage_blender: Optional[nn.Linear] = None
        self._dosage_branch_warned = False
        self._dosage_pca_mismatch_warned = False
        if self.use_dosage_branch:
            in_dim = None
            hidden2 = max(32, self.dosage_branch_hidden // 2)
            if dosage_pca_components is not None and "dosage_pca_components" not in self._buffers:
                comp = torch.as_tensor(dosage_pca_components, dtype=torch.float32)
                self.register_buffer("dosage_pca_components", comp, persistent=True)
                in_dim = int(min(comp.shape[0], comp.shape[1]))
            if dosage_pca_mean is not None and "dosage_pca_mean" not in self._buffers:
                self.register_buffer("dosage_pca_mean", torch.as_tensor(dosage_pca_mean, dtype=torch.float32), persistent=True)
            if dosage_pca_std is not None and "dosage_pca_std" not in self._buffers:
                self.register_buffer("dosage_pca_std", torch.as_tensor(dosage_pca_std, dtype=torch.float32), persistent=True)
            if in_dim is None and hasattr(self, "dosage_pca_components"):
                in_dim = int(min(self.dosage_pca_components.shape[0], self.dosage_pca_components.shape[1]))
            if in_dim is not None and self.dosage_append_realcount:
                in_dim += 1
            if in_dim is None:
                self.simple_branch = nn.Sequential(
                    nn.LazyLinear(self.dosage_branch_hidden),
                    nn.LayerNorm(self.dosage_branch_hidden),
                    nn.GELU(),
                    nn.Dropout(self.dosage_gate_dropout),
                    nn.Linear(self.dosage_branch_hidden, hidden2),
                    nn.GELU(),
                    nn.Dropout(self.dosage_gate_dropout),
                    nn.Linear(hidden2, 1)
                )
            else:
                self.simple_branch = nn.Sequential(
                    nn.Linear(in_dim, self.dosage_branch_hidden),
                    nn.LayerNorm(self.dosage_branch_hidden),
                    nn.GELU(),
                    nn.Dropout(self.dosage_gate_dropout),
                    nn.Linear(self.dosage_branch_hidden, hidden2),
                    nn.GELU(),
                    nn.Dropout(self.dosage_gate_dropout),
                    nn.Linear(hidden2, 1)
                )
            self.dosage_context_branch = nn.Sequential(
                nn.Linear(1 + self.dosage_context_dim, hidden2),
                nn.LayerNorm(hidden2),
                nn.GELU(),
                nn.Dropout(self.dosage_gate_dropout),
                nn.Linear(hidden2, 1)
            )
            self.dosage_gate = nn.Sequential(
                nn.Linear(2, self.dosage_gate_hidden),
                nn.LayerNorm(self.dosage_gate_hidden),
                nn.GELU(),
                nn.Dropout(self.dosage_gate_dropout),
                nn.Linear(self.dosage_gate_hidden, 2)
            )
            # Optional linear blender (stacking) as an alternative to softmax gate.
            self.dosage_blender = nn.Linear(2, 1)
            # Bias gate toward dosage path initially (e.g., 80/20).
            with torch.no_grad():
                for m in self.dosage_gate.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                final_lin = self.dosage_gate[-1]
                if isinstance(final_lin, nn.Linear):
                    eps = 1e-6
                    p = min(max(self.dosage_blend_prior, eps), 1.0 - eps)
                    logit = math.log(p / (1.0 - p))
                    nn.init.zeros_(final_lin.weight)
                    final_lin.bias.copy_(torch.tensor([logit, 0.0], dtype=final_lin.bias.dtype))
        elif use_dosage_branch and self.dosage_channel_idx is None:
            logging.warning("Dosage branch requested but no dosage_channel_idx provided; disabling.")

        # Temporal env encoder must be built before genomic encoder so we know env seq dim.
        env_encoder_type = str(env_encoder_type).lower()
        self.env_encoder_type = env_encoder_type
        if env_encoder_type == "lstm":
            self.env_encoder = TemporalEnvironmentalEncoder(
                n_features_per_month=n_env_features_per_month,
                n_months=n_months,
                hidden_dim=env_hidden_dim,
                num_layers=env_lstm_layers,
                env_embed_dim=env_embed_dim,
                dropout=dropout * 0.6
            )
        elif env_encoder_type == "tcn":
            self.env_encoder = TemporalConvEncoder(
                n_features_per_month=n_env_features_per_month,
                n_months=n_months,
                conv_channels=env_conv_channels,
                num_layers=env_conv_layers,
                kernel_size=env_conv_kernel,
                env_embed_dim=env_embed_dim,
                dropout=dropout * 0.6
            )
        elif env_encoder_type == "pyramid":
            pyramid_layers = env_conv_layers if env_pyramid_layers is None else int(env_pyramid_layers)
            self.env_encoder = TemporalPyramidEncoder(
                n_features_per_month=n_env_features_per_month,
                n_months=n_months,
                conv_channels=env_conv_channels,
                num_layers=pyramid_layers,
                kernel_size=env_conv_kernel,
                env_embed_dim=env_embed_dim,
                dropout=dropout * 0.6,
                scales=env_pyramid_scales
            )
        elif env_encoder_type in ("flat_mlp", "mlp"):
            self.env_encoder = FlatEnvMLPEncoder(
                n_months=n_months,
                n_features_per_month=n_env_features_per_month,
                env_embed_dim=env_embed_dim,
                hidden_dim=env_hidden_dim,
                dropout=dropout * 0.6,
                n_layers=2
            )
        else:
            raise ValueError(f"Unsupported env_encoder_type: {env_encoder_type}")
        self.use_env_anomalies = bool(use_env_anomalies)
        if env_anomaly_mean is not None:
            mean_tensor = (
                env_anomaly_mean.detach().float()
                if isinstance(env_anomaly_mean, torch.Tensor)
                else torch.as_tensor(env_anomaly_mean, dtype=torch.float32)
            )
            if mean_tensor.dim() == 3 and mean_tensor.size(0) == 1:
                mean_tensor = mean_tensor.squeeze(0)
            expected_shape = (int(n_months), int(n_env_features_per_month))
            if mean_tensor.dim() != 2 or tuple(mean_tensor.shape) != expected_shape:
                logging.warning(
                    "Env anomaly mean shape mismatch (got=%s, expected=%s); disabling anomalies.",
                    tuple(mean_tensor.shape),
                    expected_shape
                )
                self.use_env_anomalies = False
                self.env_anomaly_mean = None
            else:
                self.register_buffer("env_anomaly_mean", mean_tensor)
        else:
            self.use_env_anomalies = False
            self.env_anomaly_mean = None
        meta_embed_dim = location_embed_dim + year_embed_dim + pop_embed_dim
        if self.use_habe:
            self.genomic_encoder = None
            proj_in_dim = int(genomic_feature_dim)
            if proj_in_dim <= 0:
                raise RuntimeError("HABE input feature dim is non-positive.")
            self.genomic_feature_proj = nn.Sequential(
                nn.Linear(proj_in_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.2)
            )
            if self.use_env_film:
                self.env_to_gamma = nn.Sequential(
                    nn.Linear(env_embed_dim, embed_dim),
                    nn.Tanh()
                )
                self.env_to_beta = nn.Sequential(
                    nn.Linear(env_embed_dim, embed_dim),
                    nn.Tanh()
                )
            if self.use_env_pool_bias:
                self.env_to_pool_bias = nn.Sequential(
                    nn.Linear(env_embed_dim, embed_dim),
                    nn.Tanh()
                )
            if self.use_meta_film:
                self.meta_to_gamma = nn.Sequential(
                    nn.Linear(meta_embed_dim, embed_dim),
                    nn.Tanh()
                )
                self.meta_to_beta = nn.Sequential(
                    nn.Linear(meta_embed_dim, embed_dim),
                    nn.Tanh()
                )
            self.habe = HotspotAwareBlockEmbedding(
                feature_dim=embed_dim,
                embed_dim=embed_dim,
                max_sparse_tokens=self.max_sparse_tokens,
                num_func_types=self.num_func_types,
                use_func_type_embed=self.use_functional_token_types
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.genomic_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=max(1, int(num_intra_layers))
            )
            self.genomic_pool = AttentionPooling(embed_dim)
            # Optional env cross-attention on genomic tokens.
            self.use_env_cross_attention = bool(use_env_cross_attention)
            self.env_seq_proj = None
            self.env_cross_attention = None
            self.env_cross_norm = None
            if self.use_env_cross_attention:
                self.env_seq_proj = nn.Linear(self.env_encoder.seq_dim, embed_dim)
                self.env_cross_attention = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.env_cross_norm = nn.LayerNorm(embed_dim)
        else:
            # Chromosome-aware encoder; ignores cross-chromosome attention to preserve biological structure.
            self.genomic_encoder = ChromoAwareTransformer(
                feature_dim=genomic_feature_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=max(1, int(num_intra_layers)),
                ff_dim=ff_dim,
                dropout=dropout,
                use_pos_encoding=True,
                downsample_stride=chr_downsample_stride,
                downsample_kernel=chr_downsample_kernel,
                block_id_channel_idx=block_id_channel_idx,
                num_chromosomes=num_chromosomes,
                env_seq_dim=self.env_encoder.seq_dim,
                use_env_cross_attention=use_env_cross_attention,
                env_embed_dim=env_embed_dim,
                use_env_film=self.use_env_film,
                use_env_pool_bias=self.use_env_pool_bias,
                meta_embed_dim=meta_embed_dim,
                use_meta_film=self.use_meta_film,
                meta_film_scale=self.meta_film_scale
            )
            self.use_env_cross_attention = bool(use_env_cross_attention)

        self.location_embedding = nn.Embedding(n_locations, location_embed_dim)
        self.year_embedding = nn.Embedding(n_years, year_embed_dim)
        self.pop_embedding = nn.Embedding(n_populations, pop_embed_dim)

        fused_dim = embed_dim + env_embed_dim + location_embed_dim + year_embed_dim + pop_embed_dim
        self.fused_dim = fused_dim
        # Opt-in multi-trait auxiliary heads (hard parameter sharing) off the shared fused
        # representation. When n_aux_targets == 0 no modules are registered, so no new
        # parameters exist and forward never emits 'aux_preds' -> byte-identical.
        self.n_aux_targets = int(n_aux_targets)
        if self.n_aux_targets > 0:
            self.aux_heads = nn.ModuleList(
                [nn.Linear(self.fused_dim, 1) for _ in range(self.n_aux_targets)]
            )
        else:
            self.aux_heads = None
        self.main_head_dropout = float(dropout if main_head_dropout is None else main_head_dropout)
        self.interaction_head_dropout = float(dropout if interaction_head_dropout is None else interaction_head_dropout)
        self.fuse_dropout = nn.Dropout(self.main_head_dropout)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(self.main_head_dropout),
            nn.Linear(fused_dim, 1)
        )
        # Environment ÃƒÂ¢Ã¢â‚¬ Ã¢â‚¬â„¢ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ SNP modulation (element-wise scaling before pooling).
        self.snp_env_modulation = nn.Sequential(
            nn.Linear(env_embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.modulation_scale = nn.Parameter(torch.tensor(0.1))
        # Explicit GxE interaction path (Hadamard after projection).
        if self.low_rank_bilinear_rank > 0:
            self.interaction_bilinear = LowRankBilinear(
                embed_dim, env_embed_dim, interaction_dim, rank=self.low_rank_bilinear_rank
            )
        else:
            self.interaction_bilinear = nn.Bilinear(embed_dim, env_embed_dim, interaction_dim)
        self.g_proj = nn.Linear(embed_dim, interaction_dim)
        self.e_proj = nn.Linear(env_embed_dim, interaction_dim)
        inter_hidden = max(64, interaction_dim)
        self.interaction_head = nn.Sequential(
            nn.Linear(interaction_dim + location_embed_dim + year_embed_dim + pop_embed_dim, inter_hidden),
            nn.LayerNorm(inter_hidden),
            nn.GELU(),
            nn.Dropout(self.interaction_head_dropout),
            nn.Linear(inter_hidden, 1)
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        moe_hidden = inter_hidden if gxe_moe_hidden_dim is None else int(gxe_moe_hidden_dim)
        if self.use_gxe_moe:
            self.gxe_gate = nn.Sequential(
                nn.Linear(pop_embed_dim, max(4, pop_embed_dim)),
                nn.GELU(),
                nn.Linear(max(4, pop_embed_dim), self.gxe_moe_num_experts)
            )
            self.gxe_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(interaction_dim + location_embed_dim + year_embed_dim + pop_embed_dim, moe_hidden),
                        nn.LayerNorm(moe_hidden),
                        nn.GELU(),
                        nn.Dropout(self.interaction_head_dropout),
                        nn.Linear(moe_hidden, 1)
                    )
                    for _ in range(self.gxe_moe_num_experts)
                ]
            )
        else:
            self.gxe_gate = None
            self.gxe_experts = None
        self.fusion = GxE_FusionHead(
            embed_dim=embed_dim,
            env_embed_dim=env_embed_dim,
            location_embed_dim=location_embed_dim,
            year_embed_dim=year_embed_dim,
            pop_embed_dim=pop_embed_dim,
            interaction_dim=interaction_dim,
            dropout=dropout,
            main_head_dropout=main_head_dropout,
            interaction_head_dropout=interaction_head_dropout,
            low_rank_bilinear_rank=self.low_rank_bilinear_rank,
            interaction_reg_lambda=self.interaction_reg_lambda,
            use_gxe_moe=self.use_gxe_moe,
            gxe_moe_num_experts=self.gxe_moe_num_experts,
            gxe_moe_hidden_dim=gxe_moe_hidden_dim,
            gxe_moe_temperature=self.gxe_moe_temperature,
        )
        self.combined_dim = fused_dim + interaction_dim
        self._last_fused = None
        self._last_genomic = None
        self._last_per_chrom = None
        self._last_env = None
        self._last_env_gxe = None
        self.expects_raw_blocks = False
        self.supports_gxe_stage = True
        self._gxe_stage = 0

    @staticmethod
    def _set_module_trainable(module: Optional[nn.Module], trainable: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    def set_gxe_stage(self, stage: int) -> None:
        """
        Stage 1: train main effects only (freeze GxE path).
        Stage 2: train GxE path only (freeze main effects).
        Stage 0: train all parameters.
        """
        stage = int(stage)
        self._gxe_stage = stage
        if stage == 1:
            main_trainable, gxe_trainable = True, False
        elif stage == 2:
            main_trainable, gxe_trainable = False, True
        else:
            main_trainable, gxe_trainable = True, True

        main_modules = [
            self.env_encoder,
            self.genomic_bae,
            self.genomic_feature_proj,
            self.habe,
            self.genomic_transformer,
            self.genomic_pool,
            self.genomic_encoder,
            self.env_to_gamma,
            self.env_to_beta,
            self.env_to_pool_bias,
            self.meta_to_gamma,
            self.meta_to_beta,
            self.env_seq_proj,
            self.env_cross_attention,
            self.env_cross_norm,
            self.snp_env_modulation,
            self.location_embedding,
            self.year_embedding,
            self.pop_embedding,
            self.head,
        ]
        gxe_modules = [
            self.interaction_bilinear,
            self.g_proj,
            self.e_proj,
            self.interaction_head,
            self.gxe_gate,
            self.gxe_experts,
        ]

        for module in main_modules:
            self._set_module_trainable(module, main_trainable)
        for module in gxe_modules:
            self._set_module_trainable(module, gxe_trainable)

        # Parameters not attached to a module list above.
        if hasattr(self, "modulation_scale") and isinstance(self.modulation_scale, torch.nn.Parameter):
            self.modulation_scale.requires_grad = main_trainable
        if hasattr(self, "hotspot_focus_bias_param") and isinstance(self.hotspot_focus_bias_param, torch.nn.Parameter):
            self.hotspot_focus_bias_param.requires_grad = main_trainable
        if hasattr(self, "residual_gate") and isinstance(self.residual_gate, torch.nn.Parameter):
            self.residual_gate.requires_grad = gxe_trainable

    def _interaction_reg_penalty(self) -> Optional[torch.Tensor]:
        if self.interaction_reg_lambda <= 0.0:
            return None
        if hasattr(self.interaction_bilinear, "regularization"):
            return self.interaction_reg_lambda * self.interaction_bilinear.regularization()
        weight = getattr(self.interaction_bilinear, "weight", None)
        if weight is None:
            return None
        return self.interaction_reg_lambda * weight.abs().mean()

    @staticmethod
    def _normalize_pad_mask(mask: Optional[torch.Tensor], genomic: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert saved masks (1=real, 0=pad) or bool masks (True=pad) into bool pad masks.
        """
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            pad_mask = mask
        else:
            pad_mask = (mask == 0)
        return pad_mask.to(device=genomic.device, dtype=torch.bool)

    def _prepare_habe_inputs(
        self,
        genomic: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if genomic.dim() != 4:
            raise RuntimeError(f"Expected genomic tensor [B,C,T,F], got {tuple(genomic.shape)}")
        genomic = torch.nan_to_num(genomic, nan=0.0)
        B, C, T, F = genomic.shape
        x_flat = genomic.reshape(B, C * T, F)
        if mask is None:
            pad_mask = torch.zeros((B, C * T), device=genomic.device, dtype=torch.bool)
        else:
            pad_mask = mask.reshape(B, C * T).bool()

        # Keep block IDs unique even if upstream writer ever reuses IDs across chromosomes.
        if self.block_id_channel_idx is not None and self.block_id_channel_idx < F:
            block_raw = genomic[:, :, :, self.block_id_channel_idx].round().long()  # [B,C,T]
            # Guard against scaled/normalized block IDs: they must be integer-like for pooling to work.
            raw_vals = genomic[:, :, :, self.block_id_channel_idx]
            if not self._warned_non_integer_blocks:
                frac = (raw_vals - raw_vals.round()).abs()
                if torch.isfinite(frac).any() and frac.max() > 1e-3:
                    logging.warning("Block ID channel appears non-integer (max frac diff=%.4f); block pooling may break.", float(frac.max().item()))
                    self._warned_non_integer_blocks = True
            chrom_offset = (torch.arange(C, device=genomic.device).view(1, C, 1) + 1) * 10_000_000
            block_raw = torch.where(block_raw > 0, block_raw + chrom_offset, block_raw)
            block_raw = block_raw.reshape(B, C * T)
            block_ids = torch.where(block_raw > 0, block_raw - 1, block_raw.new_full(block_raw.shape, -1))
        else:
            pad_fill = pad_mask.new_full(pad_mask.shape, -1, dtype=torch.long)
            zeros = pad_mask.new_zeros(pad_mask.shape, dtype=torch.long)
            block_ids = torch.where(pad_mask, pad_fill, zeros)
        block_ids = block_ids.masked_fill(pad_mask, -1)

        def _safe_channel(idx: Optional[int]) -> Optional[torch.Tensor]:
            if idx is None:
                return None
            if idx < 0 or idx >= F:
                return None
            return x_flat[:, :, idx]

        def _bool_channel(idx: Optional[int]) -> torch.Tensor:
            vals = _safe_channel(idx)
            if vals is None:
                return torch.zeros_like(pad_mask, dtype=torch.bool)
            return (vals > 0.5) & (~pad_mask)

        def _high_mask(idx: Optional[int]) -> torch.Tensor:
            vals = _safe_channel(idx)
            if vals is None:
                return torch.zeros_like(pad_mask, dtype=torch.bool)
            valid = vals[~pad_mask]
            if valid.numel() == 0:
                return torch.zeros_like(pad_mask, dtype=torch.bool)
            valid = valid[torch.isfinite(valid)]
            if valid.numel() == 0:
                return torch.zeros_like(pad_mask, dtype=torch.bool)
            thresh = torch.quantile(valid.float(), self.func_hotspot_quantile)
            return (vals > thresh) & (~pad_mask)

        func_type_ids = None
        is_hotspot = torch.zeros_like(pad_mask, dtype=torch.bool)
        # [FIX] always bind these; previously assigned only when annotation channels exist,
        # which crashed the strict-hotspot step for tensors without gene/TE/promoter channels.
        is_genic = torch.zeros_like(pad_mask, dtype=torch.bool)
        is_promoter = torch.zeros_like(pad_mask, dtype=torch.bool)
        is_te = torch.zeros_like(pad_mask, dtype=torch.bool)
        hotspot_source_found = False
        use_func_context = (
            self.use_functional_hotspots
            or self.use_functional_token_types
            or self.use_functional_pool_bias
        )
        if use_func_context:
            has_type_channels = any(
                _safe_channel(idx) is not None
                for idx in (self.te_channel_idx, self.genic_channel_idx, self.promoter_channel_idx)
            )
            has_cont_channels = any(
                _safe_channel(idx) is not None
                for idx in (
                    self.block_gene_count_channel_idx,
                    self.block_snp_density_channel_idx,
                    self.block_mean_maf_channel_idx,
                )
            )
            if has_type_channels or has_cont_channels:
                is_te = _bool_channel(self.te_channel_idx) if has_type_channels else torch.zeros_like(pad_mask, dtype=torch.bool)
                is_genic = _bool_channel(self.genic_channel_idx) if has_type_channels else torch.zeros_like(pad_mask, dtype=torch.bool)
                is_promoter = _bool_channel(self.promoter_channel_idx) if has_type_channels else torch.zeros_like(pad_mask, dtype=torch.bool)
                hotspot_source_found |= has_type_channels or has_cont_channels
                te_active = True
                valid_mask = ~pad_mask
                if valid_mask.any():
                    te_mean = is_te[valid_mask].float().mean().item()
                    high = self.func_const_thresh
                    low = 1.0 - self.func_const_thresh
                    if te_mean >= high or te_mean <= low:
                        te_active = False
                        if not getattr(self, "_te_const_warned", False):
                            logging.info(
                                "TE indicator is near-constant (mean=%.3f); ignoring it for hotspots/func types.",
                                te_mean,
                            )
                            self._te_const_warned = True
                else:
                    te_active = False

                if has_type_channels:
                    func_type_ids = torch.zeros_like(pad_mask, dtype=torch.long)
                    func_type_ids = torch.where(is_genic, torch.full_like(func_type_ids, 1), func_type_ids)
                    func_type_ids = torch.where(is_promoter, torch.full_like(func_type_ids, 2), func_type_ids)
                    if te_active:
                        te_mask = is_te & (~is_genic) & (~is_promoter)
                        func_type_ids = torch.where(te_mask, torch.full_like(func_type_ids, 3), func_type_ids)
                    func_type_ids = func_type_ids.masked_fill(pad_mask, 0)

                if self.use_functional_hotspots:
                    high_maf = _high_mask(self.block_mean_maf_channel_idx) if has_cont_channels else torch.zeros_like(pad_mask, dtype=torch.bool)
                    if getattr(self, "strict_hotspots", True):
                        # Strict: only genic/promoter (optionally add high_maf by uncommenting)
                        is_hotspot = (is_genic | is_promoter) & (~pad_mask)
                        # is_hotspot = (is_genic | is_promoter | high_maf) & (~pad_mask)
                    else:
                        hot_te = is_te if te_active else torch.zeros_like(is_te, dtype=torch.bool)
                        high_gene = _high_mask(self.block_gene_count_channel_idx) if has_cont_channels else torch.zeros_like(pad_mask, dtype=torch.bool)
                        high_density = _high_mask(self.block_snp_density_channel_idx) if has_cont_channels else torch.zeros_like(pad_mask, dtype=torch.bool)
                        is_hotspot = (hot_te | is_genic | is_promoter | high_gene | high_density | high_maf) & (~pad_mask)
        elif self.te_hotspot_channel_idx is not None and self.te_hotspot_channel_idx < F:
            is_hotspot = (x_flat[:, :, self.te_hotspot_channel_idx] > 0.5) & (~pad_mask)
            hotspot_source_found = True

        if self.use_functional_hotspots and not hotspot_source_found and not getattr(self, "_warned_no_hotspot_source", False):
            logging.warning("No functional hotspot source channels found (TE/genic/promoter/block summaries); hotspots disabled.")
            self._warned_no_hotspot_source = True
        if getattr(self, "strict_hotspots", True):
            # Restrict hotspots to within blocks unless genic/promoter
            in_block = (block_ids >= 0) & (~pad_mask)
            is_hotspot = is_hotspot & (in_block | is_genic | is_promoter)

        feat = x_flat

        return feat, block_ids, is_hotspot, pad_mask, func_type_ids

    def _prepare_bae_inputs(
        self,
        x_flat: torch.Tensor,
        pad_mask: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self.use_bae or self.genomic_bae is None:
            return None
        B, L, F = x_flat.shape
        if (
            self.dosage_channel_idx is None
            or self.te_distance_channel_idx is None
            or self.gene_distance_channel_idx is None
        ):
            return None
        if (
            self.dosage_channel_idx >= F
            or self.te_distance_channel_idx >= F
            or self.gene_distance_channel_idx >= F
        ):
            return None
        dosage_raw = x_flat[:, :, self.dosage_channel_idx]
        dosage_raw = torch.nan_to_num(dosage_raw, nan=0.0)
        if float(dosage_raw.max()) <= 1.01:
            dosage_raw = dosage_raw * 2.0
        dosage_ids = torch.clamp(torch.round(dosage_raw), 0, 2).long()
        te_dist = x_flat[:, :, self.te_distance_channel_idx]
        gene_dist = x_flat[:, :, self.gene_distance_channel_idx]
        te_dist = torch.nan_to_num(te_dist, nan=0.0).clamp(min=0.0)
        gene_dist = torch.nan_to_num(gene_dist, nan=0.0).clamp(min=0.0)
        if self.distance_log1p:
            distances = torch.stack([torch.log1p(te_dist), torch.log1p(gene_dist)], dim=-1)
        else:
            distances = torch.stack([te_dist, gene_dist], dim=-1)
        hot_te = torch.zeros((B, L), device=x_flat.device, dtype=torch.float32)
        hot_prom = torch.zeros((B, L), device=x_flat.device, dtype=torch.float32)
        hot_te_idx = self.te_hotspot_channel_idx
        if hot_te_idx is None:
            hot_te_idx = self.te_channel_idx
        if hot_te_idx is not None and hot_te_idx < F:
            hot_te = (x_flat[:, :, hot_te_idx] > 0.5).float()
        if self.promoter_channel_idx is not None and self.promoter_channel_idx < F:
            hot_prom = (x_flat[:, :, self.promoter_channel_idx] > 0.5).float()
        hotspots = torch.stack([hot_te, hot_prom], dim=-1)
        if pad_mask is not None:
            dosage_ids = dosage_ids.masked_fill(pad_mask, 0)
            distances = distances.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            hotspots = hotspots.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return dosage_ids, distances, hotspots

    def _dosage_context_part(
        self,
        tensor: Optional[torch.Tensor],
        batch_size: int,
        expected_dim: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        if expected_dim <= 0:
            return torch.zeros((batch_size, 0), device=device, dtype=dtype)
        if tensor is None:
            return torch.zeros((batch_size, expected_dim), device=device, dtype=dtype)
        part = tensor
        if part.dim() == 1:
            part = part.unsqueeze(-1)
        elif part.dim() > 2:
            part = part.reshape(part.size(0), -1)
        if part.dim() != 2 or part.size(0) != batch_size:
            return torch.zeros((batch_size, expected_dim), device=device, dtype=dtype)
        if part.size(1) > expected_dim:
            part = part[:, :expected_dim]
        elif part.size(1) < expected_dim:
            part = F.pad(part, (0, expected_dim - part.size(1)))
        return part.to(device=device, dtype=dtype)

    def _dosage_branch_forward(
        self,
        genomic: torch.Tensor,
        mask: Optional[torch.Tensor],
        env_repr: Optional[torch.Tensor] = None,
        loc_emb: Optional[torch.Tensor] = None,
        year_emb: Optional[torch.Tensor] = None,
        pop_emb: Optional[torch.Tensor] = None,
        dosage_override: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if not self.use_dosage_branch or self.simple_branch is None:
            return None
        if genomic.dim() != 4:
            return None
        pad_mask = None
        if dosage_override is not None:
            flat = torch.as_tensor(dosage_override, device=genomic.device, dtype=genomic.dtype)
            if flat.dim() == 1:
                flat = flat.unsqueeze(0)
            elif flat.dim() > 2:
                flat = flat.reshape(flat.size(0), -1)
            if flat.dim() != 2 or flat.size(0) != genomic.size(0):
                return None
        else:
            if self.dosage_channel_idx is None:
                return None
            if self.dosage_channel_idx < 0 or self.dosage_channel_idx >= genomic.size(-1):
                return None
            dosage = genomic[..., self.dosage_channel_idx]
            dosage = torch.nan_to_num(dosage, nan=0.0)
            pad_mask = self._normalize_pad_mask(mask, genomic) if mask is not None else None
            flat = dosage.reshape(dosage.size(0), -1)
            if pad_mask is not None:
                pad_flat = pad_mask.reshape(dosage.size(0), -1)
                # Impute PAD positions with the mean so they become ~0 after mean-centering
                if (
                    hasattr(self, "dosage_pca_mean")
                    and self.dosage_center
                    and self.dosage_pca_mean.numel() == flat.shape[1]
                ):
                    mean = self.dosage_pca_mean.unsqueeze(0).expand_as(flat)
                    flat = torch.where(pad_flat, mean, flat)
                else:
                    flat = flat.masked_fill(pad_flat, 0.0)
        if hasattr(self, "dosage_pca_mean") and self.dosage_center:
            if self.dosage_pca_mean.numel() == flat.shape[1]:
                flat = flat - self.dosage_pca_mean
            elif not self._dosage_pca_mismatch_warned:
                logging.warning(
                    "Dosage mean length mismatch (got=%d, expected=%d); skipping mean centering.",
                    flat.shape[1], self.dosage_pca_mean.numel()
                )
                self._dosage_pca_mismatch_warned = True
        if hasattr(self, "dosage_pca_std") and self.dosage_scale:
            if self.dosage_pca_std.numel() == flat.shape[1]:
                flat = flat / torch.clamp(self.dosage_pca_std, min=1e-6)
            elif not self._dosage_pca_mismatch_warned:
                logging.warning(
                    "Dosage std length mismatch (got=%d, expected=%d); skipping std scaling.",
                    flat.shape[1], self.dosage_pca_std.numel()
                )
                self._dosage_pca_mismatch_warned = True
        if hasattr(self, "dosage_pca_components"):
            comp = self.dosage_pca_components
            if comp.dim() == 2:
                out_dim, in_dim = comp.shape
                if in_dim == flat.shape[1]:
                    flat = torch.matmul(flat, comp.t())
                elif out_dim == flat.shape[1]:
                    flat = torch.matmul(flat, comp)
                else:
                    if not self._dosage_pca_mismatch_warned:
                        logging.warning(
                            "Dosage PCA dim mismatch: input=%d, comp_shape=%s; skipping PCA.",
                            flat.shape[1], tuple(comp.shape)
                        )
                        self._dosage_pca_mismatch_warned = True
        if self.dosage_append_realcount:
            if pad_mask is not None:
                n_real = (~pad_flat).sum(dim=1, keepdim=True).to(flat.dtype)
            else:
                n_real = torch.full((flat.size(0), 1), flat.size(1), device=flat.device, dtype=flat.dtype)
            flat = torch.cat([flat, n_real], dim=1)
        env_ctx = self._dosage_context_part(
            env_repr, flat.size(0), self.env_embed_dim, flat.device, flat.dtype
        )
        loc_ctx = self._dosage_context_part(
            loc_emb, flat.size(0), self.location_embed_dim, flat.device, flat.dtype
        )
        year_ctx = self._dosage_context_part(
            year_emb, flat.size(0), self.year_embed_dim, flat.device, flat.dtype
        )
        pop_ctx = self._dosage_context_part(
            pop_emb, flat.size(0), self.pop_embed_dim, flat.device, flat.dtype
        )
        context = torch.cat([env_ctx, loc_ctx, year_ctx, pop_ctx], dim=-1)
        try:
            dosage_base = self.simple_branch(flat).squeeze(-1)
            if self.dosage_context_branch is None:
                return dosage_base
            context_in = torch.cat([dosage_base.unsqueeze(-1), context], dim=-1)
            return self.dosage_context_branch(context_in).squeeze(-1)
        except Exception as e:
            if not self._dosage_branch_warned:
                logging.warning("Dosage branch forward failed; disabling branch. Error: %s", e)
                self._dosage_branch_warned = True
            self.use_dosage_branch = False
            return None

    def _encode_genomic(
        self,
        genomic: torch.Tensor,
        mask: Optional[torch.Tensor],
        env_repr: Optional[torch.Tensor],
        env_seq: Optional[torch.Tensor],
        meta_repr: Optional[torch.Tensor],
        disable_modulation: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_habe:
            feat, block_ids, is_hotspot, pad_mask, func_type_ids = self._prepare_habe_inputs(genomic, mask)
            bae_inputs = self._prepare_bae_inputs(feat, pad_mask)
            if bae_inputs is not None and self.genomic_bae is not None:
                x = self.genomic_bae(*bae_inputs)
            else:
                x = self.genomic_feature_proj(feat)
            if (
                not disable_modulation
                and self.use_env_film
                and self.env_to_gamma is not None
                and self.env_to_beta is not None
                and env_repr is not None
            ):
                gamma = self.env_to_gamma(env_repr).unsqueeze(1)
                beta = self.env_to_beta(env_repr).unsqueeze(1)
                x = x * (1.0 + gamma) + beta
            if (
                not disable_modulation
                and self.use_meta_film
                and self.meta_to_gamma is not None
                and self.meta_to_beta is not None
                and meta_repr is not None
            ):
                meta_gamma = self.meta_to_gamma(meta_repr).unsqueeze(1) * self.meta_film_scale
                meta_beta = self.meta_to_beta(meta_repr).unsqueeze(1) * self.meta_film_scale
                x = x * (1.0 + meta_gamma) + meta_beta
            tokens, token_mask, hotspot_mask, func_type_tokens = self.habe(
                x,
                block_ids,
                is_hotspot,
                pad_mask,
                func_type_ids=func_type_ids
            )
            if not getattr(self, "_habe_debug_logged", False):
                try:
                    token_counts = (~token_mask).sum(dim=1).detach().cpu().numpy()
                    hot_counts = hotspot_mask.sum(dim=1).detach().cpu().numpy()
                    func_type_msg = ""
                    if func_type_tokens is not None:
                        flat = func_type_tokens.detach().view(-1)
                        flat = flat[flat >= 0]
                        if flat.numel() > 0:
                            counts = torch.bincount(flat, minlength=self.num_func_types).cpu().tolist()
                            func_type_msg = f" | func_type_counts={counts}"
                    logging.info(
                        "HABE tokens/sample: min=%d mean=%.1f max=%d | hotspot tokens/sample: min=%d mean=%.1f max=%d%s",
                        int(token_counts.min()),
                        float(token_counts.mean()),
                        int(token_counts.max()),
                        int(hot_counts.min()),
                        float(hot_counts.mean()),
                        int(hot_counts.max()),
                        func_type_msg,
                    )
                except Exception:
                    logging.info("HABE tokens/sample logging failed.")
                self._habe_debug_logged = True
            env_mod = None
            if not disable_modulation and env_repr is not None:
                env_mod = self.snp_env_modulation(env_repr)
                tokens = tokens * (1.0 + self.modulation_scale * env_mod.unsqueeze(1))
            if (
                not disable_modulation
                and self.use_env_cross_attention
                and env_seq is not None
                and self.env_seq_proj is not None
                and self.env_cross_attention is not None
            ):
                env_proj = self.env_seq_proj(env_seq)
                x_env, _ = self.env_cross_attention(query=tokens, key=env_proj, value=env_proj)
                tokens = self.env_cross_norm(tokens + x_env)
            encoded = self.genomic_transformer(tokens, src_key_padding_mask=token_mask)
            if env_mod is not None:
                encoded = encoded * (1.0 + self.modulation_scale * env_mod.unsqueeze(1))
            bias = None
            if self.hotspot_focus_bias > 0.0:
                stage_weight = None
                if env_seq is not None and env_seq.dim() == 3 and env_seq.size(1) > 1:
                    seq_len = env_seq.size(1)
                    start = int(0.45 * seq_len)
                    end = max(start + 1, int(0.8 * seq_len))
                    repro_slice = env_seq[:, start:end, :]
                    # Higher magnitude during reproductive stage (often heat stress) increases hotspot bias.
                    stage_energy = repro_slice.abs().mean(dim=(1, 2))
                    stage_weight = 1.0 + torch.tanh(stage_energy).unsqueeze(1)
                if stage_weight is None:
                    stage_weight = 1.0
                bias = hotspot_mask.float() * self.hotspot_focus_bias_param * stage_weight
            if (
                self.use_functional_pool_bias
                and func_type_tokens is not None
                and self.func_type_bias_weights is not None
            ):
                func_weights = self.func_type_bias_weights.to(func_type_tokens.device)
                func_bias = func_weights[func_type_tokens] * self.func_bias_scale
                bias = func_bias if bias is None else bias + func_bias
            if (
                not disable_modulation
                and self.use_env_pool_bias
                and self.env_to_pool_bias is not None
                and env_repr is not None
            ):
                env_proj = self.env_to_pool_bias(env_repr).unsqueeze(1)
                env_bias = (encoded * env_proj).sum(dim=-1) / (encoded.size(-1) ** 0.5)
                bias = env_bias if bias is None else bias + env_bias
            g_repr = self.genomic_pool(encoded, mask=token_mask, bias=bias)
            return g_repr, None

        env_mod = None
        env_seq_arg = env_seq
        env_repr_arg = env_repr
        meta_repr_arg = meta_repr
        mod_scale = self.modulation_scale
        if disable_modulation:
            env_seq_arg = None
            env_repr_arg = None
            meta_repr_arg = None
            mod_scale = None
        if env_repr is not None and not disable_modulation:
            env_mod = self.snp_env_modulation(env_repr)
        g_repr, per_chrom = self.genomic_encoder(
            genomic,
            mask=mask,
            env_seq=env_seq_arg,
            env_mod=env_mod,
            mod_scale=mod_scale,
            env_repr=env_repr_arg,
            meta_repr=meta_repr_arg
        )
        return g_repr, per_chrom

    def forward(
        self,
        genomic,
        mask,
        env_ts,
        loc_ids,
        year_ids,
        pop_ids,
        row_labels=None,
        stage: int = 0,
        return_components: bool = False,
        dosage_override: Optional[torch.Tensor] = None
    ):
        env_repr_main = self.env_encoder(env_ts)
        env_seq_main = self.env_encoder.last_sequence
        if env_seq_main is None:
            env_seq_main = torch.zeros(
                env_ts.size(0),
                self.env_encoder.n_months,
                self.env_encoder.seq_dim,
                device=env_ts.device,
                dtype=env_ts.dtype
            )
        env_repr_gxe = env_repr_main
        if self.use_env_anomalies and self.env_anomaly_mean is not None:
            env_ts_anom = env_ts - self.env_anomaly_mean
            env_repr_gxe = self.env_encoder(env_ts_anom)
        loc_emb = self.location_embedding(loc_ids)
        year_emb = self.year_embedding(year_ids)
        pop_emb = self.pop_embedding(pop_ids)
        meta_repr = torch.cat([loc_emb, year_emb, pop_emb], dim=-1)
        g_repr, per_chrom = self._encode_genomic(genomic, mask, env_repr_main, env_seq_main, meta_repr)

        if self.training and self.modality_dropout_p > 0.0:
            drop = torch.rand(1, device=g_repr.device).item()
            if drop < self.modality_dropout_p * 0.5:
                g_repr = torch.zeros_like(g_repr)
                pop_emb = torch.zeros_like(pop_emb)
            elif drop < self.modality_dropout_p:
                env_repr_main = torch.zeros_like(env_repr_main)
                env_repr_gxe = torch.zeros_like(env_repr_gxe)
                loc_emb = torch.zeros_like(loc_emb)
                year_emb = torch.zeros_like(year_emb)

        fused = torch.cat([g_repr, env_repr_main, loc_emb, year_emb, pop_emb], dim=-1)
        out_raw = self.fusion(
            g_repr,
            env_repr_main,
            loc_emb,
            year_emb,
            pop_emb,
            env_repr_gxe=env_repr_gxe,
            stage=stage,
            return_components=return_components
        )

        aux: Dict[str, torch.Tensor] = {}
        if isinstance(out_raw, (tuple, list)):
            out = out_raw[0]
            if len(out_raw) > 1 and isinstance(out_raw[1], dict):
                aux = out_raw[1]
        else:
            out = out_raw

        final_out = out
        dosage_pred = None
        dosage_weights = None
        if self.use_dosage_branch and stage == 0:
            dosage_pred = self._dosage_branch_forward(
                genomic,
                mask,
                env_repr=env_repr_main,
                loc_emb=loc_emb,
                year_emb=year_emb,
                pop_emb=pop_emb,
                dosage_override=dosage_override
            )
            complex_pred = out
            if complex_pred.dim() > 1:
                complex_pred = complex_pred.squeeze(-1)
            weight = None
            if dosage_pred is not None:
                weight = self.dosage_fixed_weight_train if self.training else self.dosage_fixed_weight_eval
                if weight is None and self.dosage_fixed_weight is not None:
                    weight = self.dosage_fixed_weight
            if dosage_pred is not None and weight is not None:
                w = torch.tensor(weight, device=dosage_pred.device, dtype=dosage_pred.dtype)
                w = torch.clamp(w, 0.0, 1.0)
                dosage_weights = torch.stack(
                    [torch.full_like(dosage_pred, w), torch.full_like(dosage_pred, 1.0 - w)],
                    dim=-1
                )
                final_out = (w * dosage_pred) + ((1.0 - w) * complex_pred)
            elif dosage_pred is not None:
                stacked = torch.stack([dosage_pred, complex_pred], dim=-1)
                if self.dosage_blender is not None:
                    final_out = self.dosage_blender(stacked).squeeze(-1)
                elif self.dosage_gate is not None:
                    gate_logits = self.dosage_gate(stacked)
                    dosage_weights = torch.softmax(gate_logits, dim=-1)
                    final_out = (dosage_weights[..., 0] * dosage_pred) + (dosage_weights[..., 1] * complex_pred)
        if dosage_weights is not None:
            aux = aux or {}
            aux["dosage_pred"] = dosage_pred
            aux["dosage_gate"] = dosage_weights
            aux["complex_pred"] = complex_pred

        if getattr(self, "aux_heads", None) is not None:
            aux = aux or {}
            aux["aux_preds"] = torch.cat([head(fused) for head in self.aux_heads], dim=1)

        self._last_fused = fused
        self._last_genomic = g_repr
        self._last_per_chrom = per_chrom
        self._last_env = env_repr_main
        self._last_env_gxe = env_repr_gxe
        if return_components or aux:
            return final_out, aux
        return final_out

    def encode_combined(self, genomic, mask, env_ts, loc_ids, year_ids, pop_ids, row_labels=None):
        """
        Returns the fused representation before the prediction head.
        """
        env_repr_main = self.env_encoder(env_ts)
        env_seq_main = self.env_encoder.last_sequence
        if env_seq_main is None:
            env_seq_main = torch.zeros(
                env_ts.size(0),
                self.env_encoder.n_months,
                self.env_encoder.seq_dim,
                device=env_ts.device,
                dtype=env_ts.dtype
            )
        env_repr_gxe = env_repr_main
        if self.use_env_anomalies and self.env_anomaly_mean is not None:
            env_ts_anom = env_ts - self.env_anomaly_mean
            env_repr_gxe = self.env_encoder(env_ts_anom)
        loc_emb = self.location_embedding(loc_ids)
        year_emb = self.year_embedding(year_ids)
        pop_emb = self.pop_embedding(pop_ids)
        meta_repr = torch.cat([loc_emb, year_emb, pop_emb], dim=-1)
        g_repr, per_chrom = self._encode_genomic(genomic, mask, env_repr_main, env_seq_main, meta_repr)
        fused = torch.cat([g_repr, env_repr_main, loc_emb, year_emb, pop_emb], dim=-1)
        self._last_per_chrom = per_chrom
        ge_feat = self.fusion.compute_ge_feat(g_repr, env_repr_gxe if env_repr_gxe is not None else env_repr_main)
        return torch.cat([fused, ge_feat], dim=-1)

    def encode_genomic_only(self, genomic, mask, row_labels=None):
        """
        Returns the un-modulated genomic representation for structure visualization.
        """
        g_repr, per_chrom = self._encode_genomic(
            genomic,
            mask,
            env_repr=None,
            env_seq=None,
            meta_repr=None,
            disable_modulation=True
        )
        self._last_genomic = g_repr
        self._last_per_chrom = per_chrom
        return g_repr

    def encode_population_only(self, pop_ids):
        """
        Returns the population embedding only (no env/genomic conditioning).
        """
        return self.pop_embedding(pop_ids)

    def encode_location_only(self, loc_ids):
        """
        Returns the location embedding only.
        """
        return self.location_embedding(loc_ids)

    def encode_year_only(self, year_ids):
        """
        Returns the year embedding only.
        """
        return self.year_embedding(year_ids)


class DualBranchGxE(nn.Module):
    """
    Wraps a GxE_Transformer_Tensor with a lightweight additive genomic branch and
    a learnable gate that blends additive vs interaction predictions. Useful when
    training multiple traits of differing complexity (simple traits can lean on
    additive signal; complex traits can lean on GxE interaction).
    """
    def __init__(
        self,
        base_model: GxE_Transformer_Tensor,
        additive_hidden_dim: int = 128,
        gate_hidden_dim: int = 32,
        gate_dropout: float = 0.1
    ):
        super().__init__()
        self.base = base_model
        self.uses_habe = getattr(base_model, "uses_habe", False)
        base_embed_dim = int(getattr(base_model, "embed_dim", additive_hidden_dim))
        self.additive_branch = nn.Sequential(
            nn.Linear(base_embed_dim, additive_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(additive_hidden_dim, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(2, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 2)
        )

    def forward(
        self,
        genomic,
        mask,
        env_ts,
        loc_ids,
        year_ids,
        pop_ids,
        row_labels=None,
        stage: int = 0,
        return_components: bool = False,
        dosage_override: Optional[torch.Tensor] = None
    ):
        # Interaction branch via the underlying transformer
        base_out = self.base(
            genomic,
            mask,
            env_ts,
            loc_ids,
            year_ids,
            pop_ids,
            row_labels=row_labels,
            stage=stage,
            return_components=True,  # always request aux to preserve components
            dosage_override=dosage_override
        )
        if isinstance(base_out, (tuple, list)):
            interaction_pred, aux = base_out
        else:
            interaction_pred, aux = base_out, {}
        if interaction_pred.dim() > 1:
            interaction_pred = interaction_pred.squeeze(-1)

        # Skip additive fusion during stage-specific residual training to preserve semantics.
        if stage == 2:
            if return_components:
                return interaction_pred, aux
            return interaction_pred

        # Additive branch: genomic-only representation -> MLP
        additive_pred = None
        try:
            g_repr = self.base.encode_genomic_only(genomic, mask, row_labels=row_labels)
            additive_pred = self.additive_branch(g_repr).squeeze(-1)
        except Exception:
            additive_pred = None

        if additive_pred is None:
            combined = interaction_pred
            weights = None
        else:
            stacked = torch.stack([additive_pred, interaction_pred], dim=-1)  # [B, 2]
            gate_logits = self.gate(stacked)
            weights = torch.softmax(gate_logits, dim=-1)
            combined = (weights[..., 0] * additive_pred) + (weights[..., 1] * interaction_pred)

        if return_components:
            aux = aux or {}
            if additive_pred is not None:
                aux["additive_out"] = additive_pred
            if weights is not None:
                aux["branch_weights"] = weights
            return combined, aux
        return combined

    # Thin proxies to keep compatibility with helper utilities (e.g., SSL pretrain).
    def encode_genomic_only(self, *args, **kwargs):
        return self.base.encode_genomic_only(*args, **kwargs)

    def encode_population_only(self, *args, **kwargs):
        return self.base.encode_population_only(*args, **kwargs)

    def encode_location_only(self, *args, **kwargs):
        return self.base.encode_location_only(*args, **kwargs)

    def encode_year_only(self, *args, **kwargs):
        return self.base.encode_year_only(*args, **kwargs)


class AdditiveGxEWrapper(nn.Module):
    """
    Adds a linear additive genomic branch in parallel to the backbone for simple traits.
    """
    def __init__(self, backbone: GxE_Transformer_Tensor):
        super().__init__()
        self.backbone = backbone
        self.uses_habe = getattr(backbone, "uses_habe", False)
        feat_dim = int(getattr(backbone, "embed_dim", 0)) or 1
        self.additive_head = nn.Linear(feat_dim, 1)

    def forward(
        self,
        genomic,
        mask,
        env_ts,
        loc_ids,
        year_ids,
        pop_ids,
        row_labels=None,
        **kwargs
    ):
        base_out = self.backbone(
            genomic,
            mask,
            env_ts,
            loc_ids,
            year_ids,
            pop_ids,
            row_labels=row_labels,
            **kwargs
        )
        meta = {}
        if isinstance(base_out, (tuple, list)):
            base_pred = base_out[0]
            if len(base_out) > 1 and isinstance(base_out[1], dict):
                meta = base_out[1]
        else:
            base_pred = base_out
        if base_pred.dim() > 1:
            base_pred = base_pred.squeeze(-1)
        g_repr = self.backbone.encode_genomic_only(genomic, mask, row_labels=row_labels)
        add_pred = self.additive_head(g_repr).squeeze(-1)
        combined = base_pred + add_pred
        if meta:
            meta = dict(meta)
            meta["additive_out"] = add_pred
            meta["base_pred"] = base_pred
            return combined, meta
        return combined

    def encode_genomic_only(self, *args, **kwargs):
        return self.backbone.encode_genomic_only(*args, **kwargs)

    def encode_population_only(self, *args, **kwargs):
        if hasattr(self.backbone, "encode_population_only"):
            return self.backbone.encode_population_only(*args, **kwargs)
        raise AttributeError("Backbone does not expose population-only encoding.")

    def encode_combined(self, *args, **kwargs):
        if hasattr(self.backbone, "encode_combined"):
            return self.backbone.encode_combined(*args, **kwargs)
        raise AttributeError("Backbone does not expose combined encoding.")

def preprocess_environmental_data(
    env_df,
    critical_features=None,
    strict: bool = True,
    n_steps: int = 20,
    engineer_features: bool = True,
    use_stage_summaries: bool = True,
    return_feature_names: bool = False,
    fit_mask: Optional[np.ndarray] = None,
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    return_stats: bool = False,
):
    """
    Preprocess environmental data into temporal format (weekly steps).
    
    Args:
        env_df: DataFrame with columns [Name, Location, Year, E_tmax_C_00, ..., E_daylength_h_19]
        critical_features: List of feature names to extract (default: 6 critical features)
        strict: When True, raise if required columns or keys are missing instead of silently
                leaving zeroed tensors.
        n_steps: Number of time steps (weeks) expected in the wide columns.
        engineer_features: When True, add derived photothermal + stress channels.
        use_stage_summaries: When True, add early/mid/late stage summary channels.
        return_feature_names: When True, return the expanded feature name list.
    
    Returns:
        env_tensor: [n_samples, n_steps, n_features_per_step]
        location_ids: [n_samples] - Integer location IDs
        year_ids: [n_samples] - Integer year IDs
        location_map: Dict mapping location names to IDs
        year_map: Dict mapping years to IDs
    """
    
    if "Location" not in env_df.columns or "Year" not in env_df.columns:
        raise ValueError("Environment dataframe must contain 'Location' and 'Year' columns.")

    if strict and "EnvKey" in env_df.columns:
        dup_env = env_df["EnvKey"].astype(str)
        dup_keys = dup_env[dup_env.duplicated()].unique()
        if len(dup_keys) > 0:
            raise ValueError(f"Duplicate EnvKey rows detected in environment data: {list(dup_keys)[:5]}")

    # Default critical features for rice flowering time
    if critical_features is None:
        critical_features = [
            'daylength_h',  # Photoperiod (MOST IMPORTANT!)
            'tmax_C',       # Max temperature
            'tmin_C',       # Min temperature
            'gdd',          # Growing degree days
            'vpd_kPa',      # Vapor pressure deficit
            'srad_allsky',  # Solar radiation
        ]

    def _has_full_wide_feature(name: str) -> bool:
        """True if all expected wide columns exist for the feature across n_steps."""
        return all(f"E_{name}_{i:02d}" in env_df.columns for i in range(n_steps))

    # Opportunistically include precipitation if wide columns exist (cumulative precip helps stage-aware modeling).
    precip_feature_name = None
    for cand in ("precip_mm", "rain_mm", "precip", "rain", "rainfall_mm"):
        if _has_full_wide_feature(cand):
            precip_feature_name = cand
            if cand not in critical_features:
                critical_features.append(cand)
                logging.info("Detected precipitation columns; adding '%s' to critical_features.", cand)
            break
    
    n_steps = int(n_steps)
    n_features = len(critical_features)
    n_samples = len(env_df)
    
    # Initialize tensor
    env_tensor = np.zeros((n_samples, n_steps, n_features), dtype=float)
    
    missing_columns = []
    # Extract features for each time step (weeks)
    for step_idx in range(n_steps):
        for feat_idx, feat_name in enumerate(critical_features):
            col_name = f'E_{feat_name}_{step_idx:02d}'
            if col_name in env_df.columns:
                env_tensor[:, step_idx, feat_idx] = env_df[col_name].values
            else:
                missing_columns.append(col_name)

    if missing_columns:
        msg = (
            f"Environment dataframe is missing {len(missing_columns)} required columns "
            f"(example: {missing_columns[:5]})."
        )
        if strict:
            raise ValueError(msg)
        print(f"Warning: {msg}")
    
    feature_names = list(critical_features)

    if engineer_features:
        feature_idx = {name: i for i, name in enumerate(critical_features)}

        def _get_feature(name: str, aliases: Tuple[str, ...] = ()) -> Optional[np.ndarray]:
            idx = feature_idx.get(name)
            if idx is None:
                for alt in aliases:
                    idx = feature_idx.get(alt)
                    if idx is not None:
                        break
            if idx is None:
                return None
            return env_tensor[:, :, idx]

        daylength = _get_feature("daylength_h", ("daylength",))
        tmax = _get_feature("tmax_C", ("tmax",))
        tmin = _get_feature("tmin_C", ("tmin",))
        gdd = _get_feature("gdd", ())
        vpd = _get_feature("vpd_kPa", ("vpd",))
        precip = _get_feature(precip_feature_name, ("precip_mm", "rain_mm", "precip", "rain", "rainfall")) if precip_feature_name else None

        required = {
            "daylength_h": daylength,
            "tmax_C": tmax,
            "tmin_C": tmin,
            "gdd": gdd,
            "vpd_kPa": vpd,
        }
        missing_req = [k for k, v in required.items() if v is None]
        if missing_req:
            raise ValueError(
                f"engineer_features=True requires {missing_req} in critical_features "
                "and corresponding columns in env data."
            )

        def _safe_percentile(arr: np.ndarray, q: float) -> float:
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return 0.0
            return float(np.percentile(flat, q * 100.0))

        heat_threshold = _safe_percentile(tmax, 0.90)
        cold_threshold = _safe_percentile(tmin, 0.10)
        vpd_threshold = _safe_percentile(vpd, 0.90)
        logging.info(
            "Auto thresholds (weekly): heat=%.3f, cold=%.3f, vpd=%.3f",
            heat_threshold, cold_threshold, vpd_threshold
        )

        photo_temp = daylength * gdd
        cum_gdd = np.cumsum(gdd, axis=1)
        cum_ptu = np.cumsum(photo_temp, axis=1)
        heat_hdd = np.maximum(0.0, tmax - heat_threshold)
        cold_cdd = np.maximum(0.0, cold_threshold - tmin)
        drought_vpd = np.maximum(0.0, vpd - vpd_threshold)
        derived_list = [photo_temp, cum_gdd, cum_ptu, heat_hdd, cold_cdd, drought_vpd]
        derived_names = [
            "photo_temp",
            "cum_gdd",
            "cum_ptu",
            "heat_hdd",
            "cold_cdd",
            "drought_vpd",
        ]
        if precip is not None:
            cum_precip = np.cumsum(precip, axis=1)
            derived_list.append(cum_precip)
            derived_names.append("cum_precip")

        derived = np.stack(derived_list, axis=-1)
        env_tensor = np.concatenate([env_tensor, derived], axis=-1)
        feature_names.extend(derived_names)

        if use_stage_summaries:
            splits = np.linspace(0, n_steps, num=4, dtype=int)
            stage_windows = [
                ("early", int(splits[0]), int(splits[1])),
                ("mid", int(splits[1]), int(splits[2])),
                ("late", int(splits[2]), int(splits[3])),
            ]
            stage_features = []
            stage_names = []
            metrics = [
                ("gdd_sum", gdd, np.sum),
                ("heat_hdd_sum", heat_hdd, np.sum),
                ("vpd_mean", vpd, np.mean),
            ]
            if precip is not None:
                metrics.append(("precip_sum", precip, np.sum))
            zeros = np.zeros((n_samples,), dtype=float)
            for stage_name, start, end in stage_windows:
                s = max(0, start)
                e = max(s, min(end, n_steps))
                for metric_name, metric_tensor, reducer in metrics:
                    seg = metric_tensor[:, s:e] if e > s else None
                    if seg is None or seg.size == 0:
                        stage_val = zeros
                    else:
                        stage_val = reducer(seg, axis=1)
                    stage_features.append(stage_val)
                    stage_names.append(f"{stage_name}_{metric_name}")
            if stage_features:
                stage_stack = np.stack(stage_features, axis=-1)
                stage_broadcast = np.repeat(stage_stack[:, None, :], n_steps, axis=1)
                env_tensor = np.concatenate([env_tensor, stage_broadcast], axis=-1)
                feature_names.extend(stage_names)

    # Standardize features
    # Default: uses provided stats or fits on the full tensor; callers should prefer train-only fit_mask to avoid leakage.
    if stats is not None:
        mean, std = stats
    else:
        fit_slice = env_tensor
        if fit_mask is not None:
            fit_mask = np.asarray(fit_mask).reshape(-1).astype(bool)
            if fit_mask.shape[0] == env_tensor.shape[0]:
                fit_slice = env_tensor[fit_mask]
            else:
                logging.warning("fit_mask length mismatch; falling back to full-env standardization.")
        mean = fit_slice.mean(axis=(0, 1), keepdims=True)
        std = fit_slice.std(axis=(0, 1), keepdims=True)
    std = std + 1e-8
    env_tensor = (env_tensor - mean) / std
    
    # Encode location and year
    location_map = {loc: idx for idx, loc in enumerate(sorted(env_df['Location'].unique()))}
    location_ids = env_df['Location'].map(location_map).values
    
    year_map = {year: idx for idx, year in enumerate(sorted(env_df['Year'].unique()))}
    year_ids = env_df['Year'].map(year_map).values
    
    if return_feature_names or return_stats:
        outs = [env_tensor, location_ids, year_ids, location_map, year_map]
        if return_feature_names:
            outs.append(feature_names)
        if return_stats:
            outs.append((mean, std))
        return tuple(outs)
    return env_tensor, location_ids, year_ids, location_map, year_map


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
