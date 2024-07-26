#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Students:
#     - ...
#     - ...

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: Why the two sentences "cats chase mice" and "mice chase cats" are considered similar using all words and sentences encoding?
	Propose a solution to fix this.

A: ...

----------------------------------------------------------

----------------------------------------------------------
Q2:  Why using concatenation, the 2nd and 4th sentences are similar?
	Can we enhance this method, so they will not be as similar?

A2: ...

----------------------------------------------------------

----------------------------------------------------------
Q3: compare between the two sentence representation, indicating their limits.

A3: ...

----------------------------------------------------------

"""

import re
import os
import sys
import math
import json
import random
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any

# ====================================================
# ============== Usefull functions ===================
# ====================================================

def vec_plus(X: List[float], Y: List[float]) -> List[float]:
    """Given two lists, we calculate the vector sum between them

    Args:
        X (List[float]): The first vector
        Y (List[float]): The second vector

    Returns:
        List[float]: The vector sum (element-wize sum)
    """
    return list(map(sum, zip(X, Y)))

def vec_divs(X: List[float], s: float) -> List[float]:
    """Given a list and a scalar, it returns another list
       where the elements are divided by the scalar

    Args:
        X (List[float]): The list to be divided
        s (float): The scalar

    Returns:
        List[float]: The resulted div list.
    """
    return list(map(lambda e: e/s, X))

# ====================================================
# ====== API: Application Programming Interface ======
# ====================================================

class WordRep:

    def fit(self, text: List[List[str]]) -> None:
        raise 'Not implemented, must be overriden'

    def transform(self, words: List[str]) -> List[List[float]]:
        raise 'Not implemented, must be overriden'

class SentRep:
    def __init__(self, word_rep: WordRep) -> None:
        self.word_rep = word_rep

    def transform(self, text: List[List[str]]) -> List[List[float]]:
        raise 'Not implemented, must be overriden'

class Sim:
    def __init__(self, sent_rep: SentRep) -> None:
        self.sent_rep = sent_rep

    def calculate(self, s1: List[str], s2: List[str]) -> float:
        raise 'Not implemented, must be overriden'
    

# ====================================================
# =========== WordRep implementations ================
# ====================================================

class OneHot(WordRep):
    def __init__(self, specials: List[str]=[]) -> None:
        super().__init__()
        #complete this
        

class TermTerm(WordRep):
    def __init__(self, window=2) -> None:
        super().__init__()
        #complete this
        
    
# ====================================================
# =========== SentRep implementations =================
# ====================================================

class ConcatSentRep(SentRep):
    def __init__(self, word_rep: WordRep, max_words: int = 10) -> None:
        super().__init__(word_rep)
        self.max_words = max_words


class CentroidSentRep(SentRep):
    pass

# ====================================================
# =========== Sim implementations ================
# ====================================================

class EuclideanSim(Sim):
    pass




# # ====================================================
# # ============ SentenceComparator class ==============
# # ====================================================

# NOT important

# ====================================================
# ===================== Tests ========================
# ====================================================

train_data = [
    ['a', 'computer', 'can', 'help', 'you'],
    ['he', 'can', 'help', 'you', 'and', 'he', 'wants', 'to', 'help', 'you'],
    ['he', 'wants', 'a', 'computer', 'and', 'a', 'computer', 'for', 'you']
]

test_data_wd = ['[S]', 'a', 'computer', 'wants', 'to', 'be', 'you']

test_data_st = [
    ['a', 'computer', 'can', 'help'],
    ['a', 'computer', 'can', 'help', 'you'],
    ['you', 'can', 'help', 'a', 'computer'],
    ['a', 'computer', 'can', 'help', 'you', 'and', 'you', 'can', 'help']
]

class DummyWordRep(WordRep):
    def __init__(self) -> None:
        super().__init__()
        self.code = {
            'a':        [1., 0., 0.],
            'computer': [0., 1., 0.],
            'can':      [0., 0., 1.],
            'help':     [1., 1., 0.],
            'you' :     [1., 0., 1.],
            'he':       [0., 1., 1.],
            'and':      [2., 0., 0.],
            'wants':    [0., 2., 0.],
            'to':       [0., 0., 2.],
            'for':      [2., 1., 0.]
        }

    def transform(self, words: List[str]) -> List[List[float]]:
        res = []
        for word in words:
            res.append(self.code[word])
        return res

def test_OneHot():
    oh_ref  = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    ohs_ref = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    
    oh_enc = OneHot()
    ohs_enc = OneHot(specials=['[CLS]', '[S]'])

    oh_enc.fit(train_data)
    ohs_enc.fit(train_data)

    print('=========================================')
    print('OneHot class test')
    print('=========================================')
    oh_res = oh_enc.transform(test_data_wd)
    print('oneHot without specials')
    print(oh_res)
    print('should be')
    print(oh_ref)

    ohs_res = ohs_enc.transform(test_data_wd)
    print('oneHot with specials')
    print(ohs_res)
    print('should be')
    print(ohs_ref)

def test_TermTerm():
    tt2_ref = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 4, 1, 0, 0, 1, 2, 1, 0, 1], 
               [4, 0, 1, 1, 1, 0, 2, 1, 0, 1], 
               [1, 1, 0, 1, 0, 2, 1, 0, 1, 0], 
               [0, 0, 0, 1, 1, 1, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 2, 3, 0, 1, 1, 0, 1, 1]]
    tt3_ref = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [2, 4, 1, 1, 1, 1, 2, 1, 0, 1], 
               [4, 2, 1, 1, 2, 1, 2, 1, 0, 1], 
               [1, 1, 0, 1, 2, 2, 2, 0, 1, 0], 
               [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [1, 2, 2, 3, 0, 2, 1, 2, 1, 1]]

    tt2_enc = TermTerm()
    tt3_enc = TermTerm(window=3)

    tt2_enc.fit(train_data)
    tt3_enc.fit(train_data)

    print('=========================================')
    print('TermTerm class test')
    print('=========================================')
    tt2_res = tt2_enc.transform(test_data_wd)
    print('TermTerm window=2')
    print(tt2_res)
    print('should be')
    print(tt2_ref)

    tt3_res = tt3_enc.transform(test_data_wd)
    print('TermTerm window=3')
    print(tt3_res)
    print('should be')
    print(tt3_ref)

def test_ConcatSentRep():
    test = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], 
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
    dummy_wd = DummyWordRep()
    concat_sent = ConcatSentRep(dummy_wd, max_words=5)
    print('=========================================')
    print('ConcatSentRep class test')
    print('=========================================')

    print(concat_sent.transform(test_data_st))
    print('must be')
    print(test)


def test_CentroidSentRep():
    test = [[0.5, 0.5, 0.25], 
            [0.6, 0.4, 0.4], 
            [0.6, 0.4, 0.4], 
            [0.7777777777777778, 0.3333333333333333, 0.4444444444444444]]
    dummy_wd = DummyWordRep()
    centroid_sent = CentroidSentRep(dummy_wd)
    print('=========================================')
    print('CentroidSentRep class test')
    print('=========================================')

    print(centroid_sent.transform(test_data_st))
    print('must be')
    print(test)

def test_Sim():
    test = [[0.5, 0.5, 0.25], 
            [0.6, 0.4, 0.4], 
            [0.6, 0.4, 0.4], 
            [0.7777777777777778, 0.3333333333333333, 0.4444444444444444]]
    dummy_wd = DummyWordRep()
    centroid_sent = CentroidSentRep(dummy_wd)
    sim = EuclideanSim(centroid_sent)
    print('=========================================')
    print('EuclideanSim class test')
    print('=========================================')

    print(centroid_sent.transform(test_data_st))
    print('must be')
    print(test)

# TODO: activate one test at once 
if __name__ == '__main__':
    test_OneHot()
    # test_TermTerm()
    # test_ConcatSentRep()
    # test_CentroidSentRep()
    # test_Sim()
    
