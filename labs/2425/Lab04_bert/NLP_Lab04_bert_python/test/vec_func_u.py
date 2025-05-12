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
import pytest
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mybert import vec_func as vf

def test_vec_ew_add():
    assert vf.vec_ew_add([1, 2], [3, 4]) == [4, 6]

def test_vec_ew_sub():
    assert vf.vec_ew_sub([5, 6], [1, 2]) == [4, 4]

def test_vec_ew_mul():
    assert vf.vec_ew_mul([2, 3], [4, 5]) == [8, 15]

def test_vec_ew_op():
    assert vf.vec_ew_op([1, 2], [3, 4], lambda x, y: x + y) == [4, 6]
    assert vf.vec_ew_op([1, 2], [3, 4], lambda x, y: x * y) == [3, 8]

def test_vec_dot():
    assert vf.vec_dot([1, 2], [3, 4]) == 11.0

def test_vec_norm():
    assert math.isclose(vf.vec_norm([3, 4]), 5.0)

def test_vec_exp():
    result = vf.vec_exp([0, 1])
    assert math.isclose(result[0], 1.0)
    assert math.isclose(result[1], math.exp(1))

def test_vec_smul():
    assert vf.vec_smul([1, 2], 3) == [3, 6]

def test_vec_shift():
    assert vf.vec_shift([3, 4], 1) == [2, 3]

def test_vec_sum():
    assert vf.vec_sum([1, 2, 3]) == 6.0

def test_vec_softmax_basic():
    result = vf.vec_softmax([1.0, 2.0])
    exp_1 = math.exp(1.0 - 2.0)
    exp_2 = math.exp(0.0)
    denom = exp_1 + exp_2
    expected = [exp_1 / denom, exp_2 / denom]
    assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(result, expected))

def test_vec_softmax_with_mask():
    result = vf.vec_softmax([1.0, 2.0], [0.0, 1.0])
    assert result[0] == 0.0
    assert math.isclose(result[1], 1.0)

def test_vec_softmax_zero_sum():
    result = vf.vec_softmax([1.0, 1.0], [0.0, 0.0])
    assert result == [0.0, 0.0]

def test_vec_concat():
    assert vf.vec_concat([1, 2], [3], [4, 5]) == [1, 2, 3, 4, 5]

def test_vec_random_length():
    result = vf.vec_random(5)
    assert len(result) == 5
    assert all(isinstance(x, float) for x in result)

def test_vec_val():
    assert vf.vec_val(3, 9) == [9, 9, 9]
    assert vf.vec_val(2, "a") == ["a", "a"]