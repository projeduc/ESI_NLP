#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a part of NLP labs, ESI, Algiers 
# --------------------------------------------------------------------
# Copyright (C) 2025 Abdelkrime Aries (kariminfo0@gmail.com)
# 
# Autors: 
#        - 2025 Abdelkrime Aries (kariminfo0@gmail.com)
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
import copy
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any

from .mat_func import mat_add_vec, mat_dot, mat_ew_op, mat_mean, mat_mean_std, mat_normalize, mat_random, mat_smul, mat_softmax, mat_transpose, mat_val
from . import Layer
from .vec_func import vec_dot, vec_ew_add, vec_ew_mul, vec_ew_op, vec_ew_sub, vec_random, vec_softmax, vec_sum, vec_val


# A linear layer used for sequences
class SeqLinear(Layer):

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
    
    def forward_single(self, X: List[List[float]]) -> List[List[float]]:
        self.X = X  # Save input for backward
        self.w_past = self.w
        Y = mat_dot(X, mat_transpose(self.w))
        if self.bias:
            Y = mat_add_vec(Y, self.b)
        return Y
    
    def forward(self, Xs: List[List[List[float]]]) -> List[List[List[float]]]:

        self.Xs = Xs  # Save input for backward
        # self.w_past = self.w

        Ys = []
        for X in Xs:
            Ys.append(self.forward_single(X))
        return Ys 

    def backward_single(self, dY: List[List[float]], alpha:float=0.01) -> List[List[float]]:
        # calculate the gradient 
        # Grad w.r.t weights: dJ/dW = dY^T @ X
        dW =  mat_dot(mat_transpose(dY), self.X)

        # Update weights
        self.w = mat_ew_op(self.w, dW, lambda w, g: w - alpha * g)

        # Update bias
        if self.bias:
            self.w = vec_ew_op(self.w, mat_transpose(dW), lambda w, G: w - alpha * vec_sum(G))

        # Gradient to pass to previous layer (dY @ W)
        return mat_dot(dY, self.w_past)  # [batch_size, in_size]
    
    def backward(self, dYs: List[List[List[float]]], alpha:float=0.01) -> List[List[List[float]]]:
        dXs = []
        for X, dY in zip(self.Xs, dYs):
            self.X = X
            dX = self.backward_single(dY, alpha=alpha)
            dXs.append(dX)
        return dXs



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
        self.X = X  # Save input for backward
        return [self.w[token] for token in X]

    def forward(self, Xs: List[List[int]]) -> List[List[List[float]]]:
        """Forward pass for a batch of sequences.

        Args:
            Xs (List[List[int]]): batch of sequences

        Returns:
            List[List[List[float]]]: embedded sequences
        """
        self.Xs = Xs  # Save input for backward
        return [self.forward_single(X) for X in Xs]

    def backward_single(self, dY: List[List[float]], alpha: float = 0.01) -> None:
        """Backward pass for a batch of sequences.

        Args:
            dY (List[List[float]]): gradients from upper layers
            alpha (float, optional): learning rate
        """

        # Update the embeddings based on the gradient from dY
        for token, grad in zip(self.X, dY):
            self.w[token] = vec_ew_op(self.w[token], grad, lambda w, g: w - alpha * g)


    def backward(self, dYs: List[List[List[float]]], alpha: float = 0.01) -> None:
        """Backward pass for a batch of sequences.

        Args:
            dYs (List[List[List[float]]]): gradients from upper layers
            alpha (float, optional): learning rate
        """

        # Update the embeddings based on the gradient from dY
        for X, dY in zip(self.Xs, dYs):
            self.X = X
            self.backward_single(dY, alpha=alpha)


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

        for x in X:
            mu, std = mat_mean_std(x, eps=self.eps)
    
            norm = mat_normalize(x, mu, std, col=True)

            self.mean.append(mu)
            self.std.append(std)
            self.norm.append(norm)
            O.append([vec_ew_add(vec_ew_mul(n, self.gamma), self.beta) for n in norm])

        return O

    def backward(self, dY: List[List[List[float]]], alpha: float = 0.01) -> List[List[List[float]]]:
        """
        dY shape: (batch_size, seq_len, features)
        """

        # Gradients w.r.t gamma and beta
        dgamma = vec_val(len(self.gamma), 0.0)
        dbeta = vec_val(len(self.beta), 0.0)

        # First pass: compute dgamma and dbeta
        for dy_batch, norm_batch in zip(dY, self.norm):
            for dy, norm in zip(dy_batch, norm_batch):
                dgamma = vec_ew_add(dgamma, vec_ew_mul(dy, norm))
                dbeta = vec_ew_add(dbeta, dy)

        # Update gamma and beta
        self.gamma = vec_ew_sub(self.gamma, [alpha * g for g in dgamma])
        self.beta = vec_ew_sub(self.beta, [alpha * b for b in dbeta])

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
        # print("SS=====", mat_dot(Q, mat_transpose(K)))
        self.S = mat_smul(mat_dot(Q, mat_transpose(K)), 1/math.sqrt(len(Q[0])))  # (T, T)
        # self.S = mat_dot(Q, mat_transpose(K))  # (T, T)
        self.P = mat_softmax(self.S, M)  # (T, T)
        Ym = mat_dot(self.P, V)  # (T, N)

        return Ym

    def forward(self, Qs: List[List[List[float]]], Ks: List[List[List[float]]], Vs: List[List[List[float]]], Ms: List[List[List[bool]]] = None) -> List[List[List[float]]]:
        """Forward for a batch of samples (M, T, d)."""
        self.Qs, self.Ks, self.Vs, self.Ms = Qs, Ks, Vs, Ms

        self.Ss = []
        self.Ps = []
        Ys      = []

        for i in range(len(Qs)):
            Ys.append(self.forward_single(Qs[i], Ks[i], Vs[i], Ms[i] if Ms else None))
            self.Ss.append(self.S)
            self.Ps.append(self.P)

        return Ys

    def backward_single(self, dY: List[List[float]], alpha: float = 0.01) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """Backward for a single sample (T, d)."""
        scale = math.sqrt(len(self.Q[0]))  # d

        dP = mat_dot(dY, mat_transpose(self.V))  # (T, T)

        if self.M is None:
            self.M = mat_val(len(self.P), len(self.P), True)

        # Softmax backward
        dS = []
        for p_r, dp_r, m_r in zip(self.P, dP, self.M):
            dot = vec_dot(p_r, dp_r)
            dp_r = [p * (dp - dot) * int(m) for p, dp, m in zip(p_r, dp_r, m_r)]
            dS.append(dp_r)

        dQ = mat_smul(mat_dot(dS, self.K), 1/scale)
        dK = mat_smul(mat_dot(mat_transpose(dS), self.Q), 1/scale)
        dV = mat_dot(mat_transpose(self.P), dY)

        return dQ, dK, dV

    def backward(self, dYs: List[List[List[float]]], alpha: float = 0.01) -> Tuple[List[List[List[float]]], List[List[List[float]]], List[List[List[float]]]]:
        """Backward for a batch of samples (M, T, d)."""
        dQs = []
        dKs = []
        dVs = []

        for i in range(len(dYs)):
            M = None if self.Ms is None else self.Ms[i]
            self.Q, self.K, self.V, self.P, self.M = self.Qs[i], self.Ks[i], self.Vs[i], self.Ps[i], M
            dQ, dK, dV = self.backward_single(dYs[i], alpha=alpha)
            dQs.append(dQ)
            dKs.append(dK)
            dVs.append(dV)

        return dQs, dKs, dVs



