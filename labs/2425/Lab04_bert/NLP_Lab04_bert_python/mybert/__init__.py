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

import pickle

class Module:
    def forward(self, Xs):
        """Forward pass: should return output"""
        raise NotImplementedError

    def backward(self, dYs, alpha):
        """Backward pass: should return gradient w.r.t. input"""
        raise NotImplementedError

class Layer(Module):
    def forward_single(self, X):
        """Forward pass: should return output"""
        raise NotImplementedError

    def backward_single(self, dY, alpha):
        """Backward pass: should return gradient w.r.t. input"""
        raise NotImplementedError
    

class Model(Module):
    def save(self, location: str) -> None:
        """Save the model to a file."""
        with open(location, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(location: str) -> 'Model':
        """Load the model from a file."""
        with open(location, 'rb') as f:
            return pickle.load(f)