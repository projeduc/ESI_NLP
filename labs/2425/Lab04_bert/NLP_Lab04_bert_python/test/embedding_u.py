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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mybert.layers import Embedding

emb = Embedding(5, 3)

emb.w = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.2, 0.1],
        [0.1, 0.1, 0.4],
        [0.2, 0.1, 0.1]
    ]

X = [[0, 1], [3, 2]]

Y = [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [[0.1, 0.1, 0.4], [0.3, 0.2, 0.1]]]

dY = [[[-1, 1, 1], [-0.5, 0.5, -0.5]], [[1, 1, 1], [2, 1, 2]]]

nw = [
        [0.2, 0.1, 0.19999999999999998], 
        [0.25, 0.25, 0.45], 
        [0.09999999999999998, 0.1, -0.1], 
        [0.0, 0.0, 0.30000000000000004], 
        [0.2, 0.1, 0.1]
        ]


def test_Embedding():

    assert len(emb.w) == 5 # vocab

    assert len(emb.w[0]) == 3 # embedding size

    assert emb.forward(X) == Y

    emb.backward(dY, alpha=0.1)

    assert nw == emb.w