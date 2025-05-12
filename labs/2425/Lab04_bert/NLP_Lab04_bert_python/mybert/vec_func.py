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
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any


def vec_ew_add(X: List[float], Y: List[float]) -> List[float]:
    """Element-wise addition: X + Y"""
    return list(map(sum, zip(X, Y)))

def vec_ew_sub(X: List[float], Y: List[float]) -> List[float]:
    """Element-wise subtraction: X - Y"""
    return list(map(lambda x_y: x_y[0] - x_y[1], zip(X, Y)))

def vec_ew_mul(X: List[float], Y: List[float]) -> List[float]:
    """Element-wise multiplication: X * Y"""
    return list(map(lambda x_y: x_y[0] * x_y[1], zip(X, Y)))

def vec_ew_op(X: List[float], Y: List[float], op) -> List[float]:
    """Applies a binary operation element-wise on two vectors X and Y"""
    return list(map(lambda x_y: op(x_y[0], x_y[1]), zip(X, Y)))

def vec_dot(X: List[float], Y: List[float]) -> float:
    return reduce(lambda acc, x_y: acc + x_y[0] * x_y[1], zip(X, Y), 0.0)

def vec_norm(X: List[float]) -> float:
    """Computes the L2 norm of vector X"""
    return math.sqrt(reduce(lambda acc, x: acc + x*x, X, 0.0))

def vec_exp(X: List[float]) -> List[float]:
    """Applies exponential to each element in a vector: e^X"""
    return list(map(math.exp, X))

def vec_smul(X: List[float], s: float) -> List[float]:
    """Multiplies a vector by a scalar: X * s"""
    return list(map(lambda e: e * s, X))

def vec_shift(X: List[float], v: float) -> List[float]:
    """Shifts each element in a vector by v"""
    return list(map(lambda x: x - v, X))

def vec_sum(X: List[float]) -> float:
    """Sum of all elements in the vector X"""
    return reduce(lambda acc, x: acc + x, X, 0.0)


def vec_softmax(X: List[float], M: List[bool]=None) -> List[float]:
    r = vec_exp(vec_shift(X, max(X)))
    if M is not None:
        r = vec_ew_op(r, M, lambda x, y: x * int(y))
    s = vec_sum(r)
    if s == 0.0:
        return [0.0] * len(X)  # Avoid division by zero

    return vec_smul(r, 1/s)

def vec_concat(*vectors: List[float]) -> List[float]:
    """Concatenates many vectors to one"""
    return reduce(lambda acc, X: acc + X, vectors, [])


def vec_random(n: int) -> List[float]:
    """Creates a vectors of n lmnts"""
    return [random.random() for _ in range(n)]

def vec_val(n: int, v: Any) -> List[Any]:
    """Creates a vectors of n lmnts"""
    return [v] * n

