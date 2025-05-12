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

import os, sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mybert.layers import SeqLinear

slin = SeqLinear(3, 4)

# slin.w = [
#     [0.1, 0.2, 0.3, 0.4],
#     [0.5, 0.6, 0.7, 0.8],
#     [0.2, 0.1, 0.0, 0.3]
# ]

slin.w = [
    [0.1, 0.5, 0.2],
    [0.2, 0.6, 0.1],
    [0.3, 0.7, 0. ],
    [0.4, 0.8, 0.3]
]

X = [
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[2, 1, 2], [3, 3, 3], [2, 0, 1]]
]

Yt = [
    [
        [ 1.7,  1.7,  1.7,  2.9],
        [ 4.1,  4.4,  4.7,  7.4],
        [ 6.5,  7.1,  7.7, 11.9]
    ],
    [
        [ 1.1,  1.2,  1.3,  2.2],
        [ 2.4,  2.7,  3. ,  4.5],
        [ 0.4,  0.5,  0.6,  1.1]
    ]
]

dY = [
    [
        [ 2,  -1,  1,  2],
        [ 1,  1,  1,  1],
        [ -1,  2,  0, 2]
    ],
    [
        [ 3,  -1,  -3,  2],
        [ 2,  2,  3 ,  4],
        [ 1,  2,  2,  1]
    ]
]

dXt = [
    [[1.1, 2.7, 0.9], [1., 2.6, 0.6], [1.1, 2.3, 0.6]],
    [[0., 0.4, 1.1], [3.1, 7.5, 1.8], [1.5, 3.9, 0.7]]
    ]


def test_SeqLinear_cr():

    lin = SeqLinear(6, 4)

    assert len(lin.w) == 4 # out

    assert len(lin.w[0]) == 6 # in


def test_SeqLinear_fb():
    Y = slin.forward(X)
    
    assert all(all(a == pytest.approx(b) for a, b in zip(y, yt)) for y, yt in zip(Y, Yt))

    dX = slin.backward(dY, alpha=0.1)

    assert all(all(a == pytest.approx(b) for a, b in zip(dx, dxt)) for dx, dxt in zip(dX, dXt))

    