#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: 
PoS: We note that the vector representing our input is too sparse 
(many zeros) which causes a bigger model.
Propose a solution.
----------------------------------------------------------
A1:

----------------------------------------------------------
Q2:
PoS: We note that the model is not really accurate.
It is far from being acceptable.
Propose a solution to enhance MEMM prediction.
----------------------------------------------------------
A2:

----------------------------------------------------------
Q3:
CKY: What is the benefit and the downfall of using PoS in 
CKY parsing instead of a hand-prepared lexicon?
----------------------------------------------------------
A3:

----------------------------------------------------------
Q4:
CKY: Can we change CKY to handle unitary productions (A --> B)? 
If yes, how? If no, why?
----------------------------------------------------------
A4: 

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

def build_parse_tree(T: List[List[List[Tuple[str, int, int, int]]]], 
                     sent: List[str], 
                     i: int, 
                     j: int, 
                     pos: int) -> Tuple[str, Any, Any]:
    A, k, iB, iC = T[i][j][pos]
    if k >= 0:
        left = build_parse_tree(T, sent, i, k, iB)
        right = build_parse_tree(T, sent, k+1, j, iC)
        return (A, left, right)
    return (A, sent[i])

def bool2int(b: bool) -> int:
    return 1 if b else 0

# ====================================================
# =============== Global variables ===================
# ====================================================

prefixes = [
    "ambi", "anti", "astro", "bi", "co", "con", "de", "dis", "em", "extra", "fore" "hetero"
    "hind", "homo", "im", "in", "inter", "mal", "mid", "mis", "mono", "non", "on", "pan",
    "ped", "post", "pre", "pro", "re", "semi", "sub", "sur", "trans", "tri", "twi", "ultra",
    "un", "uni", "under", "up"
]

suffixes = [
    "able", "ac", "ize", "age", "al", "an", "ant", "ary", "cracy", "cycle", "dom", "eer", "en",
    "er", "ess", "est", "ette", "ful", "hood", "ible", "ic", "ify", "ion", "ish", "ism", "ity",
    "less", "like", "log", "ment", "ness", "or", "ous", "ship", "th", "ure", "ward", "wise", "y"
]

# ====================================================
# ================== MEMM class ======================
# ====================================================

# TODO complete this function
def encode_word(word: str) -> List[float]:
    return None


class MEMM:

    def __init__(self, tag_list: List[str]):
        """_summary_

        Args:
            tag_list (List[str]): A list of tags list.
        """

        self.tag_list = tag_list

        # Recover the length of the encoding input
        nb_feat = len(self.encode('.', '.', '.', '.'))

        # each tag is represented by a maxent function
        self.perceptron = [
            [random.random() for i in range(nb_feat+1)] for j in range(len(tag_list))]
        
    #TODO complete encoding function
    def encode(self, pword: str, ptag: str, cword: str, fword: str) -> List[float]:
        """Encoding the input of MEMM.
        It encodes the past words and tags, the current word, and the future words 
        into one vector.

        Args:
            pword (str): the past word.
            ptag (str): the past tag.
            cword (str): the current word.
            fword (str): the future word.

        Returns:
            List[float]: vector encoding of these components based on their features.
        """
        
        return None
    
    def __predict(self, input: List[float]) -> List[float]:
        """This is a private method
        It takes a vector representation of whatever the input is,
        then it returns the logits (probabilities) of tags

        Args:
            input (List[float]): The input representation.

        Returns:
            List[float]: The logits; tags' probabilities.
        """
        res = []
        s = 0.0
        for tag_weights in self.perceptron:
            output = reduce(lambda c, e: c + e[0] * e[1], zip(input, tag_weights), 0.0)
            output = math.exp(output)
            s += output
            res.append(output)
        return [e/s for e in res]

    def predict(self, pword: str, ptag: str, cword: str, fword: str, cls:bool=True) -> Union[str,list[float]]:
        """Returns either the logits or the tag.

        Args:
            pword (str): the past word.
            ptag (str): the past tag.
            cword (str): the current word.
            fword (str): the future word.
            cls (bool, optional): If we want the tag [True] or the logits [False] as output. Defaults to True.

        Returns:
            Union[str,list[float]]: Either the tag or the logits.
        """
        # Always add the bias before the encoded vector
        input = [1.0] + self.encode(pword, ptag, cword, fword)
        # Get the logits
        res = self.__predict(input)
        # return either the winning tag or the loggits
        return self.tag_list[res.index(max(res))] if cls else res

    
    def fit_step(self, data: List[List[Tuple[str, str]]], alpha:float=0.01) -> None:
        """Trains our MEMM on one epoch.

        Args:
            data (List[List[Tuple[str, str]]]): A list of sentences.
            Each sentence is encoded as a list of tuples (word, tag)

            alpha (float, optional): Learning rate. Defaults to 0.01.
        """
        M = 0
        dJ = [[0.0] * len(self.perceptron[0]) for i in range(len(self.perceptron))]
        for sent in data:
            pword, ptag = '<s>', '<t>'
            slen = len(sent)
            M += slen
            for i in range(slen):
                cword, ctag = sent[i]
                fword = '</s>' if i+1 >= slen else sent[i+1][0]
                # Always add the bias 
                X = [1.0] + self.encode(pword, ptag, cword, fword)
                H = self.__predict(X)
                Y = [0.0] * len(H)
                Y[self.tag_list.index(ctag)] = 1.0
                for ci in range(len(H)):
                    grad = H[ci] - Y[ci]
                    for xi in range(len(X)):
                        dJ[ci][xi] += grad * X[xi]
                pword, ptag = cword, ctag

        # Update weights
        alpha = alpha / M
        for ci in range(len(self.perceptron)):
            for fi in range(len(self.perceptron[0])):
                self.perceptron[ci][fi] -= alpha * dJ[ci][fi]
        
    def fit(self, data: List[List[Tuple[str, str]]], alpha=0.01, max_it:int=100):
        """Trains our MEMM on a set of tagged sentences.

        Args:
            data (List[List[Tuple[str, str]]]): A list of sentences.
            Each sentence is encoded as a list of tuples (word, tag)

            alpha (float, optional): Learning rate. Defaults to 0.01.
            max_it (int, optional): Maximum of iterations. Defaults to 100.
        """
        for it in range(max_it):
            self.fit_step(data, alpha=alpha)

    def predict_all(self, sent: List[str]) -> List[str]:
        """Predicts all the tags of a given sentence.
        In this case, we use greedy decoding 
        (we do not try to maximize the overall probability).

        Args:
            sent (List[str]): A tokenized sentence.

        Returns:
            List[str]: A list of tags.
        """
        tags = []
        pword, ptag = '<s>', '<t>'
        slen = len(sent)
        for i, word in enumerate(sent):
            fword = '</s>' if i+1 >= slen else sent[i+1]
            tag = self.predict(pword, ptag, word, fword)
            tags.append(tag)
            pword, ptag = word, tag
        return tags

# ====================================================
# ================== CKY class ======================
# ====================================================

class CKY:
    def __init__(self, gram: List[Tuple[str, str, str]], lex: MEMM):
        # Grammaire
        self.gram = gram
        self.lex = lex

    # TODO Complete CKY function
    def parse(self, sent: List[str]) -> List[List[List[Tuple[str, int, int, int]]]]:
        T = []
        N = len(sent)
        return T

# ====================================================
# ================== Parser class ====================
# ====================================================

class Parser:

    def fit_lexer(self, url: str) -> None:
        f = open(url, 'r', encoding='utf8')
        data = []
        tags = set()
        for l in f: # Reading each line
            if len(l) < 5: # if the length is less than 5, we ignore it 
                continue
            # preprocess the sentence
            sent = l.strip(' \t\r\n').replace('\/', '\\').split()
            p = []
            data.append(p)
            for word_tag in sent:
                word, tag = word_tag.split("/")
                p.append((word.replace('\\', '/'), tag))
                tags.add(tag)
        f.close()

        self.lex = MEMM(list(tags))
        self.lex.fit(data)

    def fit_parser(self, url: str) -> None:
        f = open(url, 'r', encoding='utf8')
        if not self.lex:
            raise Exception('You have to train a lexer first; use fit_lexer')
        
        gram = []
        for l in f: # Reading line by line
            l = l.strip(' \t\r\n')
            if len(l) < 5 or l.startswith('#'):
                continue
            info = l.split('\t')
            if len(info) == 3:
                gram.append((info[0], info[1], info[2]))
        f.close()

        self.cky = CKY(gram, self.lex)

    def parse(self, sentence: str) -> Tuple[str, Any, Any]:
        tokens = sentence.split()
        n = len(tokens) - 1
        T = self.cky.parse(tokens)

        for pos in range(len(T[0][n])):
            if T[0][n][pos][0] == 'S':
                return build_parse_tree(T, tokens, 0, n, pos)
        raise Exception('No parse tree for the sentence: ' + sentence)


# def tester_viterbi():
#     annotateur = MorphoSyntaxe()
#     annotateur.entrainer('./data/univ_train.txt')
#     phrase = 'this has some logic .'
#     print('viterbi doit être comme gourmand')
#     print('force brute : ', annotateur.estimer(phrase))
#     print('gourmand : ', annotateur.estimer(phrase, opt='gourmand'))
#     print('viterbi : ', annotateur.estimer(phrase, opt='viterbi'))


# ====================================================
# ===================== Tests ========================
# ====================================================

# def evaluation():
#     annotateur = MorphoSyntaxe()
#     annotateur.entrainer('./data/univ_train.txt')
#     annotateur.charger_evaluation('./data/pen_test.txt')
#     annotateur.evaluer(-1)

def test_encode_word():
    test = {
        '<s>': (79, 0),
        'comparable': (79, 2),
        'uncomparable': (79, 2),
    }
    print('=======================================')
    print('Testing "encode_word" function ')
    print('=======================================')
    for t in test:
        tt = test[t]
        code = encode_word(t)
        print('---------------------------')
        print('word:', t, ', size:', tt[0], ', sum of ones:', tt[1])
        print('your size:', len(code), ', your sum of ones:', sum(code))

def test_MEMM_encode():
    test = {
        ('<s>', '<t>', 'I', 'teach'): ([1, 0, 0, 0, 0], 0),
        ('I', 'B', 'teach', 'NLP'): ([0, 0, 1, 0, 0], 0),
    }
    r = MEMM(['A', 'B', 'C', 'D'])
    print('=======================================')
    print('Testing "MEMM.encode" method')
    print('=======================================')
    for t in test:
        tt = test[t]
        code = r.encode(*t)
        print('---------------------------')
        print('arguments:', t)
        print('Five first elements:', tt[0])
        print('Your 5 first elements:', code[:5])

def test_MEMM():
    parser = Parser()
    parser.fit_lexer('./data/pos_test.txt')
    memm = parser.lex
    sentence = ['he', 'can', 'help']
    print('5 classes')
    print(memm.predict('<s>', '<t>', 'he', 'can', cls=False))
    print(memm.predict_all(sentence))

class DummyLex:
    def predict_all(self, sent: List[str]) -> List[str]:
        res = []
        for word in sent:
            if word in ['he', 'she', 'they', 'you']:
                res.append('PR')
                continue
            if word in ['form', 'forms', 'fish']:
                if len(res) == 0 or res[-1] in ['PR', 'NN']:
                    res.append('VB')
                else:
                    res.append('NN')
                continue 
            if word in ['a', 'the']:
                res.append('DT')
                continue 
            if word == 'eats':
                res.append('VB')
                continue 
            if word == 'cat':
                res.append('NN')
                continue
            
            # default: adjective
            res.append('AJ')
        return res
      

def test_CKY():
    parser = Parser()
    parser.lex = DummyLex()
    parser.fit_parser('./data/gram1.txt')
    sent = 'the fish form a little form'
    result = "('S', ('NP', ('DT', 'the'), ('NN', 'fish')), ('VP', ('VB', 'form'), ('NP', ('DT', 'a'), ('AP', ('AJ', 'little'), ('NN', 'form')))))"
    tree = parser.parse(sent)
    print('Expected result:', result)
    print('My result: ', tree)
    try:
        _ = parser.parse('the cat eats the mouse')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    test_encode_word()
    # test_MEMM_encode()
    # test_MEMM()
    # test_CKY()
