#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a paart of NLP labs, ESI, Algiers 
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


import math
import json
import random
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any

from mat_func import mat_add_vec, mat_dot, mat_ew_op, mat_mean, mat_mean_std, mat_normalize, mat_random, mat_smul, mat_softmax, mat_transpose, mat_val
from . import Layer
from vec_func import vec_dot, vec_ew_add, vec_ew_mul, vec_ew_op, vec_ew_sub, vec_random, vec_softmax, vec_sum, vec_val


# since there is no sequence in herem no need to implement forward_single
# mostly, it is forward function that uses forward_single (for simplification)
# but, in here, we'll do the inverse
class Linear(Layer):

    def __init__(self, in_size: int, out_size: int, bias: bool = False) -> None:
        """A Linear layer with many neurons.
        Args:
            in_size (int): the number of features in input (do not count bias)
            out_size (int): the number of neurons (the output)
            bias (bool): is there a bias or not
        """
        self.w = mat_random(out_size, in_size)
        self.bias = bias
        if bias:
            self.b = vec_random(out_size)
    
    def forward_single(self, X: List[float]) -> List[float]:
        """Given some input, predict the output (one sample)

        Args:
            X (List[float]): Encoded input

        Returns:
            List[float]: The output
        """
        return self.forward([X])[0]
    
    def forward(self, Xs: List[List[float]]) -> List[List[float]]:
        """Given some input, predict the output (many samples)

        Args:
            Xs (List[List[float]]): Encoded input

        Returns:
            List[List[float]]: The output
        """
        self.Xs = Xs  # Save input for backward
        Ys = mat_dot(Xs, self.w)
        if self.bias:
            Ys = mat_add_vec(Ys, self.b)
        return Ys 

    def backward_single(self, dY: List[float], alpha:float=0.01) -> None:
        """Updates the weights of the neurons

        Args:
            dY (List[float]): List of losts 
            alpha (float, optional): update factor. Defaults to 0.01.
        """
        return self.backward([dY], alpha=alpha)
    
    def backward(self, dYs: List[List[float]], alpha:float=0.01) -> None:
        """Updates the weights of the neurons

        Args:
            dYs (List[List[float]]): List of losts 
            alpha (float, optional): update factor. Defaults to 0.01.
        """
        # normalize ( no need, It is supposed the gradients are normalized in the loss function)
        # alpha /= len(self.X)

        # calculate the gradient 
        # Grad w.r.t weights: dJ/dW = dY^T @ X
        dW =  mat_dot(mat_transpose(dYs), self.X)

        # Update weights
        self.w = mat_ew_op(self.w, dW, lambda w, g: w - alpha * g)

        # Update bias
        if self.bias:
            self.w = vec_ew_op(self.w, mat_transpose(dW), lambda w, G: w - alpha * vec_sum(G))

        # Gradient to pass to previous layer (dY @ W)
        return mat_dot(dYs, self.w)  # [batch_size, in_size]




# ====================================================
# ======================= TODO =======================
# ====================================================

class Embedding(Layer):

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """An Embedding layer mapping tokens to vectors."""
        self.w = mat_random(vocab_size, embed_dim)

    def forward_single(self, X: List[int]) -> List[List[float]]:
        """Forward pass for one sequence.

        Args:
            X (List[int]): one sequence

        Returns:
            List[List[float]]: embedded sequence
        """
        return None

    def forward(self, Xs: List[List[int]]) -> List[List[List[float]]]:
        """Forward pass for a batch of sequences.

        Args:
            Xs (List[List[int]]): batch of sequences

        Returns:
            List[List[List[float]]]: embedded sequences
        """
 
        return None

    def backward_single(self, dY: List[List[float]], alpha: float = 0.01) -> None:
        """Backward pass for a batch of sequences.

        Args:
            dY (List[List[float]]): gradients from upper layers
            alpha (float, optional): learning rate
        """

        # Update the embeddings based on the gradient from dY



    def backward(self, dYs: List[List[List[float]]], alpha: float = 0.01) -> None:
        """Backward pass for a batch of sequences.

        Args:
            dYs (List[List[List[float]]]): gradients from upper layers
            alpha (float, optional): learning rate
        """

        # Update the embeddings based on the gradient from dY





class LayerNorm(Layer):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = vec_val(dim, 1.0)  # shape: (features,)
        self.beta = vec_val(dim, 0.0)   # shape: (features,)

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        X shape: (batch_size, seq_len, features)
        """
        self.X = X

        self.mean, self.std, self.norm = [], [], []

        O = []

        return O

    def backward(self, dY: List[List[List[float]]], alpha: float = 0.01) -> List[List[List[float]]]:
        """
        dY shape: (batch_size, seq_len, features)
        """

        # Gradients w.r.t gamma and beta
        dgamma = vec_val(len(self.gamma), 0.0)
        dbeta = vec_val(len(self.beta), 0.0)

        # First pass: compute dgamma and dbeta
        

        # Update gamma and beta
        

        # Second pass: compute dX
        dX = []
        for x_batch, dy_batch, mean_batch, std_batch in zip(self.X, dY, self.mean, self.std):
            dX_batch = []

            for x, dy, mu, std in zip(x_batch, dy_batch, mean_batch, std_batch):
                dx_norm = vec_ew_mul(dy, self.gamma)

                dvar = vec_sum([
                    dx_norm[j] * (x[j] - mu) * -0.5 / (std ** 3)
                    for j in range(len(x))
                ])

                dmean = vec_sum([
                    -dx_norm[j] / std for j in range(len(x))
                ]) + dvar * vec_sum([
                    -2 * (x[j] - mu) for j in range(len(x))
                ]) / len(x)

                dx = [
                    dx_norm[j] / std +
                    dvar * 2 * (x[j] - mu) / len(x) +
                    dmean / len(x)
                    for j in range(len(x))
                ]
                dX_batch.append(dx)

            dX.append(dX_batch)

        return dX



class ScaledDotProductAttention(Layer):
    def __init__(self) -> None:
        pass  # No learnable parameters

    def forward_single(self, Q: List[List[float]], K: List[List[float]], V: List[List[float]], M: List[List[bool]] = None) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """Forward for a single sample (T, d)."""

        return None

    def forward(self, Qs: List[List[List[float]]], Ks: List[List[List[float]]], Vs: List[List[List[float]]], Ms: List[List[List[bool]]] = None) -> List[List[List[float]]]:
        """Forward for a batch of samples (M, T, d)."""
        self.Qs, self.Ks, self.Vs, self.Ms = Qs, Ks, Vs, Ms

        self.Ss = []
        self.Ps = []
        Ys      = []

        return Ys

    def backward_single(self, dY: List[List[float]], alpha: float = 0.01) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """Backward for a single sample (T, d)."""

        dQ = None
        dK = None
        dV = None

        return dQ, dK, dV

    def backward(self, dYs: List[List[List[float]]], alpha: float = 0.01) -> Tuple[List[List[List[float]]], List[List[List[float]]], List[List[List[float]]]]:
        """Backward for a batch of samples (M, T, d)."""
        dQs = []
        dKs = []
        dVs = []

        return dQs, dKs, dVs


