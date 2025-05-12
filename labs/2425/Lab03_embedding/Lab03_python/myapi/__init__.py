#!/usr/bin/env python
# -*- coding: utf-8 -*-



class Module:
    def forward(self, X):
        """Forward pass: should return output"""
        raise NotImplementedError

    def backward(self, dJ):
        """Backward pass: should return gradient w.r.t. input"""
        raise NotImplementedError

class Layer(Module):
    pass