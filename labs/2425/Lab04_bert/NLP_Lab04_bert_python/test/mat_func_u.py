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

from mybert import mat_func as mu

@pytest.fixture
def sample_matrices():
    X = [[1.0, 2.0], [3.0, 4.0]]
    Y = [[5.0, 6.0], [7.0, 8.0]]
    return X, Y

def test_mat_ew_add(sample_matrices):
    X, Y = sample_matrices
    assert mu.mat_ew_add(X, Y) == [[6.0, 8.0], [10.0, 12.0]]

def test_mat_ew_sub(sample_matrices):
    X, Y = sample_matrices
    assert mu.mat_ew_sub(Y, X) == [[4.0, 4.0], [4.0, 4.0]]

def test_mat_ew_mul(sample_matrices):
    X, Y = sample_matrices
    assert mu.mat_ew_mul(X, Y) == [[5.0, 12.0], [21.0, 32.0]]

def test_mat_dot(sample_matrices):
    X, Y = sample_matrices
    assert mu.mat_dot(X, Y) == [[19.0, 22.0], [43.0, 50.0]]

def test_mat_add_vec():
    X = [[1, 2], [3, 4]]
    v = [10, 20]
    assert mu.mat_add_vec(X, v) == [[11, 12], [23, 24]]
    assert mu.mat_add_vec(X, v, col=True) == [[11, 22], [13, 24]]
    

def test_mat_flatten():
    X = [[1, 2], [3, 4]]
    assert mu.mat_flatten(X) == [1, 2, 3, 4]
    assert mu.mat_flatten(X, col=True) == [1, 3, 2, 4]
    

def test_mat_transpose():
    X = [[1, 2], [3, 4]]
    assert mu.mat_transpose(X) == [[1, 3], [2, 4]]

def test_mat_sum():
    X = [[1, 2], [3, 4]]
    assert mu.mat_sum(X, col=True) == [4, 6]
    assert mu.mat_sum(X, col=False) == [3, 7]

def test_mat_mean():
    X = [[1, 2], [3, 4]]
    assert mu.mat_mean(X) == [1.5, 3.5]
    assert mu.mat_mean(X, col=True) == [2.0, 3.0]

def test_mat_mean_var():
    X = [[1, 2, 3], [3, 4, 5]]
    means, vars = mu.mat_mean_var(X)
    assert means == [2, 4]
    assert all(abs(v - vt) < 0.05 for v, vt in zip(vars, [0.67, 0.67]))

    means, vars = mu.mat_mean_var(X, col=True)
    assert means == [2, 3, 4]
    assert all(abs(v - vt) < 0.05 for v, vt in zip(vars, [1, 1, 1]))

def test_mat_mean_std():
    X = [[1, 2, 3], [3, 4, 5]]
    means, std = mu.mat_mean_std(X)
    assert means == [2, 4]
    assert all(abs(v - vt) < 0.05 for v, vt in zip(std, [0.82, 0.82]))

    means, vars = mu.mat_mean_std(X, col=True)
    assert means == [2, 3, 4]
    assert all(abs(v - vt) < 0.05 for v, vt in zip(vars, [1, 1, 1]))

def test_mat_normalize():
    X = [[1, 2, 3], [3, 4, 5]]
    mu2 = [1.5, 3.5] # not the real one
    std2 = [0.5, 0.5] # not the real one
    mu1 = [1.5, 2.5, 4.5] # not the real one
    std1 = [0.5, 0.25, 0.75] # not the real one

    result = mu.mat_normalize(X, mu1, std1)
    assert all(all(abs(a - b) < 0.01 for a, b in zip(r, e)) for r, e in zip(result, [[-1, -2, -2], [3, 6, 0.67]]))

    result = mu.mat_normalize(X, mu2, std2, col=True)
    assert all(all(abs(a - b) < 0.01 for a, b in zip(r, e)) for r, e in zip(result, [[-1, 1, 3], [-1, 1, 3]]))

def test_mat_smul():
    X = [[1, 2], [3, 4]]
    assert mu.mat_smul(X, 2) == [[2, 4], [6, 8]]

def test_mat_val():
    assert mu.mat_val(2, 3, 9) == [[9, 9, 9], [9, 9, 9]]

def test_mat_random_shape():
    mat = mu.mat_random(3, 4)
    assert len(mat) == 3
    assert all(len(row) == 4 for row in mat)
