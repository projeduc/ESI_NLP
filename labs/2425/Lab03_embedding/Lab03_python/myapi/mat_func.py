
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is a paart of NLP labs, ESI, Algiers 
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
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any
from .vec_func import vec_dot, vec_ew_add, vec_ew_mul, vec_ew_op, vec_ew_sub, vec_random, vec_smul, vec_softmax, vec_val


def mat_ew_add(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """Element-wise matrix addition: X + Y"""
    return [vec_ew_add(row_x, row_y) for row_x, row_y in zip(X, Y)]

def mat_ew_sub(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """Element-wise matrix subtraction: X - Y"""
    return [vec_ew_sub(row_x, row_y) for row_x, row_y in zip(X, Y)]

def mat_ew_mul(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """Element-wise matrix multiplication: X * Y"""
    return [vec_ew_mul(row_x, row_y) for row_x, row_y in zip(X, Y)]

def mat_ew_op(X: List[List[float]], Y: List[List[float]], op) -> List[List[float]]:
    """Applies a binary operation element-wise on two matrices A and B"""
    return [vec_ew_op(row_x, row_y, op) for row_x, row_y in zip(X, Y)]

def mat_dot(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication between X and Y"""
    # Transpose Y for easy column access
    Y_T = list(zip(*Y))

    # Multiply rows of X with columns of Y (Y_T)
    return [[vec_dot(row_x, col_y) for col_y in Y_T] for row_x in X]


def mat_add_vec(X: List[List[float]], v: List[float], col: bool = False) -> List[List[float]]:
    """
    Adds a vector to each row (default) or column (if col=True) of a matrix X.

    Args:
        X (List[List[float]]): Matrix to add to
        v (List[float]): Vector to add
        col (bool): If True, add vector to columns; else to rows

    Returns:
        List[List[float]]: Resulting matrix after addition
    """
    if col:
        # Add v[j] to each element in column j
        return [[X[i][j] + v[j] for j in range(len(v))] for i in range(len(X))]
    
    # Add v[i] to each element in row i
    return [[X[i][j] + v[i] for j in range(len(X[0]))] for i in range(len(v))]


def mat_flatten(X: List[List[float]], col: bool = False) -> List[float]:
    """
    Flattens a 2D matrix into a 1D list.

    Args:
        X (List[List[float]]): The matrix to flatten.
        col (bool): If True, flattens column-wise; otherwise row-wise.

    Returns:
        List[float]: The flattened vector.
    """
    X2 = zip(*X) if col else X
    return [x for X1 in X2 for x in X1]

def mat_random(n: int, m:int) -> List[List[float]]:
    """Creates a matrix of n,m lmnts"""
    return [vec_random(m) for _ in range(n)]

def mat_val(n: int, m: int, v: Any) -> List[List[Any]]:
    """Creates a matrix of size (n × m) filled with value v"""
    return [vec_val(m, v) for _ in range(n)]




def mat_transpose(X: List[List[float]]) -> List[List[float]]:
    """Transposes a matrix X (n x m) into (m x n)"""
    return [list(row) for row in zip(*X)]


def mat_sum(X: List[List[float]], col: bool = False) -> List[float]:
    """Sums the matrix X over rows (default) or columns.

    Args:
        X (List[List[float]]): The matrix.
        col (bool, optional): If True, sum over columns. If False, sum over rows. Defaults to False.

    Returns:
        List[float]: The summed vector.
    """
    X2 = zip(*X) if col else X
    return [sum(V) for V in X2]

def mat_mean(X: List[List[float]], col: bool = False) -> List[float]:
    """Averages the matrix X over rows (default) or columns.
    The columns must have the same size
    Args:
        X (List[List[float]]): The matrix.
        col (bool, optional): If True, average over columns. If False, average over rows. Defaults to False.

    Returns:
        List[float]: The averaged vector.
    """
    if col:
        X2, l = zip(*X), len(X)
    else:
        X2, l = X, len(X[0])
    return [sum(V) / l for V in X2]

def mat_mean_var(X: List[List[float]], col: bool = False) -> Tuple[List[float], List[float]]:
    """Computes the mean and variance of the matrix X over rows (default) or columns.

    Args:
        X (List[List[float]]): The matrix.
        col (bool, optional): If True, compute over columns. If False, over rows. Defaults to False.

    Returns:
        Tuple[List[float], List[float]]: A tuple (means, variances).
    """
    if col:
        X2, l = zip(*X), len(X)
    else:
        X2, l = X, len(X[0])
    
    means = [sum(V) / l for V in X2]
    vars_ = [sum((x - mean)**2 for x in V) / l for V, mean in zip(X2, means)]
    
    return means, vars_

def mat_mean_std(X: List[List[float]], col: bool = False, eps: float = 1e-5) -> Tuple[List[float], List[float]]:
    """Computes the mean and standard deviation of the matrix X over rows (default) or columns,
    adding a small epsilon for numerical stability.

    Args:
        X (List[List[float]]): The matrix.
        col (bool, optional): If True, compute over columns. If False, over rows. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.

    Returns:
        Tuple[List[float], List[float]]: A tuple (means, std_devs).
    """
    if col:
        X2, l = zip(*X), len(X)
    else:
        X2, l = X, len(X[0])

    means = [sum(V) / l for V in X2]
    stds = [math.sqrt(sum((x - mean)**2 for x in V) / l + eps) for V, mean in zip(X2, means)]

    return means, stds

def mat_normalize(X: List[List[float]], mu: List[float], std: List[float], col: bool = False, eps: float = 1e-5) -> List[List[float]]:
    """Normalizes the matrix M given mean (mu) and std (std) for each row (default) or column.

    Args:
        X (List[List[float]]): The matrix.
        mu (List[float]): The means.
        std (List[float]): The standard deviations.
        col (bool, optional): If True, normalize over columns. If False, over rows. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.

    Returns:
        List[List[float]]: The normalized matrix.
    """
    if col:
        X = list(zip(*X))  # transpose if normalizing over columns

    X_norm = [[(x - m) / (s + eps) for x, m, s in zip(row, mu, std)] for row in X]


    if col:
        X_norm = list(map(list, zip(*X_norm)))  # transpose back

    return X_norm

def mat_smul(X: List[List[float]], s: float) -> List[List[float]]:
    """Multiplies a vector by a scalar: X * s"""
    return list(map(lambda x: vec_smul(x, s), X))



def mat_softmax(X: List[List[float]], M: List[List[bool]]=None, col: bool = False) -> List[List[float]]:
    if col:
        X = list(zip(*X))  # transpose if normalizing over columns
    if M is not None:
        return list(map(lambda x_m: vec_softmax(x_m[0], x_m[1]), X, M))
    
    return list(map(lambda x: vec_softmax(x), X))