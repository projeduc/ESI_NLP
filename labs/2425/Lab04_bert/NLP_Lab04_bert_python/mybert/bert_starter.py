#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a part of NLP labs, ESI, Algiers 
# --------------------------------------------------------------------
# Copyright (C) 2025 Abdelkrime Aries (kariminfo0@gmail.com)
# 
# Autors: 
#        - 2025 Abdelkrime Aries (kariminfo0@gmail.com)
# Contributors:
#        - 
#        - 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from typing import List, Tuple
from . import Module, Model
from .mat_func import mat_ew_add
from .layers import ScaledDotProductAttention, SeqLinear, LayerNorm, Embedding

# ==========================================================================================================
# ============================================ Useful functions ============================================
# ==========================================================================================================

def split_heads(X: List[List[List[float]]], num_heads: int, head_dim: int) -> List[List[List[List[float]]]]:
    """
    Split input of shape [batch, seq_len, dim_model] into
    [num_heads, batch, seq_len, head_dim]
    """
    batch_size = len(X)
    seq_len = len(X[0])
    split = [[] for _ in range(num_heads)]  # num_heads groups

    for b in range(batch_size):
        for t in range(seq_len):
            token = X[b][t]
            for h in range(num_heads):
                start = h * head_dim
                end = start + head_dim
                if len(split[h]) <= b:
                    split[h].append([])  # Add batch index
                split[h][b].append(token[start:end])
    return split



def merge_heads(X: List[List[List[List[float]]]]) -> List[List[List[float]]]:
    """
    Merge from [num_heads, batch, seq_len, head_dim] to
    [batch, seq_len, dim_model]
    """
    num_heads = len(X)
    batch_size = len(X[0])
    seq_len = len(X[0][0])
    
    merged = []
    for b in range(batch_size):
        seq = []
        for t in range(seq_len):
            merged_token = []
            for h in range(num_heads):
                merged_token += X[h][b][t]
            seq.append(merged_token)
        merged.append(seq)
    return merged

def construct_mask(mask: List[List[bool]]) -> List[List[List[bool]]]:
    """
    Constructs a square attention mask per sample:
    result[i][j][k] = mask[i][j] and mask[i][k]

    Args:
        mask (List[List[bool]]): Batch of 1D token visibility masks.

    Returns:
        List[List[List[bool]]]: Batch of 2D attention masks.
    """
    if mask is None:
        return None
    result = []
    for sample in mask:
        sample_mask = []
        result.append(sample_mask)
        for i in range(len(sample)):
            if sample[i]:
                row = sample[:]  # shallow copy: reuse booleans
            else:
                row = [False] * len(sample)
            sample_mask.append(row)
    return result

def tensor3_ew_add(X: List[List[List[float]]], Y: List[List[List[float]]]) -> List[List[List[float]]]:
    """Element-wise addition: X + Y"""
    return [mat_ew_add(row_x, row_y) for row_x, row_y in zip(X, Y)]


# ==========================================================================================================
# ======================================= Your suffering begins here =======================================
# ==========================================================================================================


class MultiheadAttention(Module):
    def __init__(self, dim_model: int, num_heads: int) -> None:
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads

        # Use Linear layers for projections
        self.q_proj = SeqLinear(dim_model, dim_model)
        self.k_proj = SeqLinear(dim_model, dim_model)
        self.v_proj = SeqLinear(dim_model, dim_model)
        self.o_proj = SeqLinear(dim_model, dim_model)

        # One attention layer per head
        self.attentions = [ScaledDotProductAttention() for _ in range(num_heads)]

    # TODO: complete
    def forward(self, 
                query: List[List[List[float]]], 
                key  : List[List[List[float]]], 
                value: List[List[List[float]]], 
                mask : List[List[List[bool]]] = None
                ) -> List[List[List[float]]]:
        """Forward pass."""

    
        return None
    

    # TODO: complete
    def backward(self, 
                 dY: List[List[List[float]]], 
                 alpha: float=0.01
                 ) -> Tuple[List[List[List[float]]], List[List[List[float]]], List[List[List[float]]]]:
        """Backward pass."""

        dQuery = None
        dKey = None
        dValue = None

        return dQuery, dKey, dValue
    


class BertBlock(Module):
    def __init__(self, dim_model: int, num_heads: int, dim_ff: int) -> None:
        self.attention = MultiheadAttention(dim_model, num_heads)
        self.norm1 = LayerNorm(dim_model)

        self.ffn1 = SeqLinear(dim_model, dim_ff)
        self.ffn2 = SeqLinear(dim_ff, dim_model)

        self.norm2 = LayerNorm(dim_model)

    # TODO: complete
    def forward(self, X: List[List[List[float]]], M: List[List[List[bool]]] = None) -> List[List[List[float]]]:
        

        return None

    # TODO: complete
    def backward(self, dY: List[List[List[float]]], alpha: float = 0.01) -> List[List[List[float]]]:
        # dY: [batch_size, seq_len, dim_model]

        return None


class Bert(Model):
    def __init__(self, vocab_size: int, max_len: int, dim_model: int, num_heads: int, dim_ff: int, num_layers: int, segment_size: int = 2) -> None:
        # Embeddings
        self.token_embedding    = Embedding(vocab_size, dim_model)
        self.position_embedding = Embedding(max_len, dim_model)
        self.segment_embedding  = Embedding(segment_size, dim_model)

        # Transformer blocks
        self.blocks = [BertBlock(dim_model, num_heads, dim_ff) for _ in range(num_layers)]

    # TODO: complete
    def forward(
        self,
        token_ids  : List[List[int]],               # token IDs: [batch, seq_len]
        segment_ids: List[List[int]],               # segment IDs: [batch, seq_len]
        mask       : List[List[List[bool]]] = None  # attention mask: [batch, seq_len, seq_len]
    ) -> List[List[List[float]]]:
        
        batch_size, seq_len = len(token_ids), len(token_ids[0]) 

        # Position indices
        pos_ids = [[i for i in range(seq_len)] for _ in range(batch_size)]

        # Masks
        M = construct_mask(mask)


        return None

    # TODO: complete
    def backward(self, dY: List[List[List[float]]], alpha: float = 0.01) -> None:
        pass


