#!/usr/bin/env python
# -*- coding: utf-8 -*-


# First student: .....
# Second student: ....

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: - Why the two sentences "cats chase mice" and "mice chase cats" are considered similar 
    using both encodings (TF and CBOW)?
    - Propose a solution to fix this.
----------------------------------------------------------
A1:
- Because ...

- Solution: ...

----------------------------------------------------------
Q2: Why "cats chase mice" and "I fish" are having some similarity 
    using CBOW ?

----------------------------------------------------------
A2: Because ...

----------------------------------------------------------
Q3: "mice consume fish" and "rodents eat sardines" are quit similar.
    But using CBOW, they are not that similar.
    - Why?
    - Propose a different model which reflects this similarity.

----------------------------------------------------------
A3: 
   - Because ...
   - Model: ...

"""

import math
import json
import random
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any

# ====================================================
# ============== Usefull functions ===================
# ====================================================

def onehot(lst: List[Any], e: Any) -> List[int]:
    """Given a list of elements and an element,
    this function returns a onehot ending
    of this element according to the list

    Args:
        lst (List[Any]): a list of elements.
        e (Any): the element to be encoded.

    Returns:
        List[int]: OneHot encoding of the element "e" based on the list "lst".
    """
    res = [0] * len(lst)
    if e in lst:
        res[lst.index(e)] = 1
    return res

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

def get_sentences(url: str) -> List[List[str]]:
    """Given a URL, returns a list of sentences, each is a list of words.
    The file is textual where each line contains a sentence

    Args:
        url (str): The path of the file

    Returns:
        List[List[str]]: A list of tokenized sentences 
    """
    f = open(url, 'r', encoding='utf8')
    data: List[List[str]] = []
    for l in f: # Reading each line
        if len(l) < 5: # if the length is less than 5, we ignore it 
            continue
        # preprocess the sentence and add it into the data
        data.append(l.strip(' \t\r\n').split())
            
    f.close()
    return data


class Perceptron:

    def __init__(self, size: int, nb_feat: int, softmax: bool = False) -> None:
        """A perceptron is a layer with many neurons.
        Here, we have two possible activation functions:
        - Softmax
        - Unit : no activation at all

        Args:
            size (int): the number of neurons (the output)
            nb_feat (int): the number of features in input (do not count bias)
            softmax (bool, optional): using softmax or not. Defaults to False.
        """
        self.w = [[random.random() for i in range(nb_feat+1)] for j in range(size)]
        self.softmax = softmax
        
    def predict(self, X: List[float]) -> List[float]:
        """Given some input, predict the output (one sample at once)

        Args:
            X (List[float]): Encoded input

        Returns:
            List[float]: The output
        """
        result = []
        s = 0.0
        for w in self.w:
            output = reduce(lambda c, e: c + e[0] * e[1], zip(X, w), 0.0)
            if self.softmax:
                output = math.exp(output)
                s += output
            result.append(output)
        if self.softmax:
            return [e/s for e in result] 
        return result
    
    def predict_all(self, X: List[List[float]]) -> List[List[float]]:
        """Predicts outputs for many inputs.

        Args:
            X (List[List[float]]): A list of encoded inputs.

        Returns:
            List[List[float]]: A list of outputs.
        """
        return [self.predict(x) for x in X]
    
    def update_weights(self, A: List[List[float]], delta: List[List[float]], alpha:float=0.01) -> None:
        """Updates the weights of the neurons

        Args:
            A (List[List[float]]): List of activations (inputs)
            delta (List[List[float]]): List of losts 
            alpha (float, optional): update factor. Defaults to 0.01.
        """
        alpha /= len(A)

        # calculate the gradient 
        dJ = [[0.0] * len(self.w[0]) for i in range(len(self.w))]
        for i in range(len(A)):
            for oi in range(len(self.w)):
                dJ[oi][0] = delta[i][oi]
                for ii in range(len(self.w[0])-1):
                    dJ[oi][ii+1] += delta[i][oi] * A[i][ii]

        # Update weights
        for oi in range(len(self.w)):
            for ii in range(len(self.w[0])):
                self.w[oi][ii] -= alpha * dJ[oi][ii]



# ====================================================
# =============== SentEncoder class ==================
# ====================================================

class SentEncoder:
    def __init__(self, terms: List[str]) -> None:
        self.terms = terms

    def encode(self, sent: List[str]) -> List[float]:
        raise 'Not implemented' 

# ====================================================
# ================ TFEncoder class ===================
# ====================================================

class TFEncoder(SentEncoder):
    
    # TODO: complete TF encoding
    def encode(self, sent: List[str]) -> List[float]:

        return None

# ====================================================
# =============== CBOWEncoder class ==================
# ====================================================

class CBOWEncoder(SentEncoder):
    def __init__(self, terms: List[str], emb_size:int = 5) -> None:
        super().__init__(terms)
        self.emb_size = emb_size

        self.train_terms = ['<s>', '</s>'] + terms

        # Number of features
        nb_feat: int = len(self.train_terms) * 2 # one before and one after

        # TODO: First layer 
        self.enc: Perceptron = None

        # TODO: Second layer
        self.dec: Perceptron = None

        # Storing each terms encoding
        self.embedding = {}
        

    # TODO: complete data preprocessing
    def encode_samples(self, sents: List[List[str]]) -> Tuple[List[List[int]], List[List[int]], List[str]]:
        X = []
        Y = []
        L = []       
        return X, Y, L
    
    def fit_step(self, X: List[List[float]], Y: List[List[float]]) -> None:
        dJ1: List[List[float]] = []
        dJ2: List[List[float]] = []
        for x in X:
            dJ1.append([0.0] * self.emb_size)
            dJ2.append([0.0] * len(Y[0]))

        o1 = self.enc.predict_all(X)
        o2 = self.dec.predict_all(o1)

        for i in range(len(Y)):
            for j in range(len(Y[0])):
                dJ2[i][j] = o2[i][j] - Y[i][j]
                for k in range(self.emb_size):
                    dJ1[i][k] += dJ2[i][j] * self.dec.w[j][k]
            for k in range(self.emb_size):
                    dJ1[i][k] /= len(Y[0])

        self.dec.update_weights(o1, dJ2)
        self.enc.update_weights(X, dJ1)


    def fit(self, samples: List[List[str]], max_it:int=100) -> None:
        X, Y, L = self.encode_samples(samples)

        for it in range(max_it):
            self.fit_step(X, Y)

        E = self.enc.predict_all(X)
        for i in range(len(L)):
            self.embedding[L[i]] = E[i]


    # TODO: complete sentence encoding
    def encode(self, sent: List[str]) -> List[float]:
        return None

# ====================================================
# ============ SentenceComparator class ==============
# ====================================================

class SentenceComparator:

    def __init__(self, data: List[List[str]], TF: bool=False, param: Dict[str, Any]={}) -> None:

        self.encoder: SentEncoder = None

        # calculate the different terms
        terms = set()
        for sent in data:
            for term in sent:
                terms.add(term)
    
        if TF:
            self.encoder = TFEncoder(list(terms))
        else:
            self.encoder = CBOWEncoder(list(terms), **param)
            self.encoder.fit(data)

    # TODO: complete cosine similarity
    def compare(self, s1: List[str], s2: List[str]) -> float:
        return None

# ====================================================
# ================ Plagiarism class ==================
# ====================================================

class Plagiarism:
    def __init__(self) -> None:
        self.sent_comp: Dict[str, SentenceComparator] = {}

    def fit_all(self, url: str) -> None:
        data = get_sentences(url)
        # fit a TF encoder 
        self.sent_comp['TF'] = SentenceComparator(data, TF=True)
        # fit a CBOW encoder
        self.sent_comp['CBOW'] = SentenceComparator(data, TF=False)

    def compare(self, url1:str, url2: str) -> None:
        data1 = get_sentences(url1)
        data2 = get_sentences(url2)
        for sent1 in data1:
            for sent2 in data2:
                print('-----------------------------------')
                print(sent1, ' = ', sent2, ' ?')
                for name, sent_cmp in self.sent_comp.items() :
                    print(name, ' :', sent_cmp.compare(sent1, sent2))

# ====================================================
# ===================== Tests ========================
# ====================================================

def test_TFEncoder_encode():
    tests = [
        ('A B C D A A', [3, 1, 1]),
        ('A A B B B D E E', [2, 3, 0]),
        ('D D D', [0, 0, 0])
    ]

    tf_enc = TFEncoder(['A', 'B', 'C'])

    print('=========================================')
    print('TFEncoder encode method test')
    print('=========================================')
    for test in tests:
        print('TF(' + test[0] + ') =', tf_enc.encode(test[0].split()), 'must be', test[1])


def test_CBOWEncoder_encode_samples():
    sents = [['A', 'A', 'B'], ['B', 'A']]
    X_true = [
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0]
    ]
    Y_true = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]
    L_true = ['A', 'A', 'B', 'B', 'A']

    cbow_enc = CBOWEncoder(['A', 'B'])

    print('=========================================')
    print('CBOWEncoder __encode_samples method test')
    print('=========================================')

    X, Y, L = cbow_enc.encode_samples(sents)

    print(X, 'must be', X_true)
    print(Y, 'must be', Y_true)
    print(L, 'must be', L_true)


def test_CBOWEncoder_encode():
    
    cbow_enc = CBOWEncoder(['A', 'B'], emb_size=2)
    sents = [['A', 'A', 'B'], ['B', 'A']]
    cbow_enc.fit(sents)

    A_code = cbow_enc.embedding['A']
    B_code = cbow_enc.embedding['B']

    AAA_code = A_code
    BB_code = B_code
    ABBA_code = list(map(lambda e: e/2, vec_plus(A_code, B_code)))
    ABBB_code = list(map(lambda e: e * 3, B_code))
    ABBB_code = vec_plus(ABBB_code, A_code)
    ABBB_code = vec_divs(ABBB_code, 4)

    print('=========================================')
    print('CBOWEncoder __encode_samples method test')
    print('=========================================')

    print('A A A', cbow_enc.encode(['A', 'A', 'A']), 'must be', AAA_code)
    print('B B', cbow_enc.encode(['B', 'B']), 'must be', BB_code)
    print('A B B A', cbow_enc.encode(['A', 'B', 'B', 'A']), 'must be', ABBA_code)
    print('A B B B', cbow_enc.encode(['A', 'B', 'B', 'B']), 'must be', ABBB_code)
    print('A B C B A', cbow_enc.encode(['A', 'B', 'C', 'B', 'A']), 'must be', ABBA_code)
    print('C C', cbow_enc.encode(['C', 'C']), 'must be', [0.0, 0.0])


def test_SentenceComparator_compare():
    data = [['A', 'A', 'B'], ['B', 'A']]
    s_c_tf = SentenceComparator(data, TF=True)
    s_c_cbow = SentenceComparator(data, TF=False)

    tests = [
        (['A', 'B'], ['A', 'A'], 0.7071067811865475, 0.9505948599072084),
        (['D'], ['C'], 0, 0)
    ]

    print('=========================================')
    print('SentenceComparator compare method test')
    print('=========================================')

    for test in tests:
        print('-------------------')
        print(test[0], test[1])
        print('TF', s_c_tf.compare(test[0], test[1]), 'must be', test[2])
        print('CBOW', s_c_cbow.compare(test[0], test[1]), 'must be', test[3])



def test_plagiarism():
    plagiarism = Plagiarism()
    plagiarism.fit_all('./data/train.txt')
    plagiarism.compare('./data/test1.txt', './data/test2.txt')


# Activate one test at once 
if __name__ == '__main__':
    test_TFEncoder_encode()
    # test_CBOWEncoder_encode_samples()
    # test_CBOWEncoder_encode()
    # test_SentenceComparator_compare()
    # test_plagiarism()

    
