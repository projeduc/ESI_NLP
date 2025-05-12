#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Submit just this file, no zip, no additional files
# -------------------------------------------------

# Students:
#     - ...
#     - ...

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: "If PoS tagging is used with CKY, there will be no ambiguities". True or False? Justify.

A1: ...

----------------------------------------------------------

----------------------------------------------------------
Q2: If PoS tagging is used with CKY, there will be no out-of-vocabulary problem". True or False? Justify.

A2: ...

----------------------------------------------------------
"""

import re
import random
from typing import Dict, List, Tuple, Set


# ======================================================
#                   Analyse CKY
# ======================================================

class CKY:
    def __init__(self, gram: List[Tuple[str, str, str]], lex: Dict[str, List[str]]):
        # Grammar
        self.gram = gram # list((A, B, C)| (A, B)) : A -> B C | A -> B
        self.lex = lex # word --> liste(PoS)

    # TODO: Complete this method
    def parse(self, sent: List[str]) -> List[List[List[Tuple[str, int, int, int]]]]:
        T = []
        N = len(sent)
        for i in range(N):
            T.append([[] for j in range(N)])
            word = sent[i]
            for A in self.lex[word]: # The word is supposed to exist in the lexicon
                T[i][i].append((A, -1, -1, -1)) # All possible leaves are added
            
        # Complete here

        return T

    def export_json(self):
        return self.__dict__.copy()

    def import_json(self, data):
        for key in data:
            self.__dict__[key] = data[key]

# ======================================================
#                Parsing
# ======================================================

# TODO: complete this function
def pr_eval(ref, sys):
    return 0., 0.

def construct(T, sent, i, j, pos):
    A, k, iB, iC = T[i][j][pos]
    #A = A.upper()
    if k >= 0:
        left = construct(T, sent, i, k, iB)
        if iC == -1:
            return (A, left)
        right = construct(T, sent, k+1, j, iC)
        return (A, left, right)
    return (A, sent[i])

def parse_tuple(string):
    string = re.sub(r'([^\s(),]+)', "'\\1'", string)
    #print(string)
    try:
        s = eval(string)
        if type(s) == tuple:
            return s
        return
    except:
        return

class Syntax():
    def __init__(self):
        self.eval = []

    def _parse(self, sent):
        r = None
        T = self.model.parse(sent)
        n = len(sent) - 1
        # for i in range(n+1):
        #     for j in range(i, n+1):
        #         print(i+1, j+1, T[i][j])
        for pos in range(len(T[0][n])):
            if T[0][n][pos][0] == 'S':
                # print('bingo')
                r = construct(T, sent, 0, n, pos)
                break
        return r

    def parse(self, sent: str):
        return self._parse(sent.strip().lower().split())

    def load_model(self, url):
        f = open(url, 'r')
        lex = {}
        gram = []
        for l in f: # line-by-line reading
            l = l.strip()
            if len(l) < 3 or l.startswith('#') :
                continue
            info = l.split('	')
            if len(info) == 2:
                if info[1][0].isupper():
                    gram.append((info[0], info[1]))
                else:
                    if not info[1] in lex:
                        lex[info[1]] = []
                    lex[info[1]].append(info[0])
            elif len(info) == 3:
                gram.append((info[0], info[1], info[2]))
        self.model = CKY(gram, lex)
        f.close()


    def load_eval(self, url):
        f = open(url, 'r')
        for l in f: # line-by-line reading
            l = l.strip()
            if len(l) < 5 or l.startswith('#'):
                continue
            info = l.split('	')
            
            self.eval.append((info[0], parse_tuple(info[1])))

    def evaluate(self, n):
        if n == -1:
            S = self.eval
            n = len(S)
        else :
            S = random.sample(self.eval, n)
        P, R = 0.0, 0.0
        for i in range(n):
            test = S[i]
            print('=======================')
            print('sent:', test[0])
            print('ref tree:', test[1])
            tree = self.parse(test[0])
            print('sys tree:', tree)
            P_i, R_i = pr_eval(test[1], tree)
            print('P, R:', P_i, R_i)
            P += P_i
            R += R_i

        P, R = P/n, R/n
        print('---------------------------------')
        print('P, R: ', P, R)



# ======================================================
#        Graph generation (Dot language)
# ======================================================

def generate_node(node, id=0):
    # If the node does not exist
    if node is None:
        return 0, ''
    # If the node is final
    nid = id + 1
    if (len(node) == 2) and (type(node[1]) == str) :
        return nid, 'N' + str(id) + '[label="' + node[0] + "=" + node[1] + '" shape=box];\n'
    # Otherzise,
    # If there are children, print if else
    res = 'N' + str(id) + '[label="' + node[0] + '"];\n'
    nid_l = nid
    nid, code = generate_node(node[1], id=nid_l)
    res += code
    res += 'N' + str(id) + ' ->  N' + str(nid_l) + ';\n'
    if len(node) > 2:
        nid_r = nid
        nid, code = generate_node(node[2], id=nid_r)
        res += code
        res += 'N' + str(id) + ' ->  N' + str(nid_r) + ';\n'
    return nid, res

def generate_graphviz(root, url):
    res = 'digraph Tree {\n'
    id, code = generate_node(root)
    res += code
    res += '}'
    f = open(url, 'w')
    f.write(res)
    f.close()


# ======================================================
#                       TESTES
# ======================================================

# Test in the lecture
def test_cky():
    parser = Syntax()
    parser.load_model('data/gram1.txt')
    sent = 'la petite forme une petite phrase'
    result = "('S', ('NP', ('D', 'la'), ('N', 'petite')), ('VP', ('V', 'forme'), ('NP', ('D', 'une'), ('AP', ('J', 'petite'), ('N', 'phrase')))))"
    tree = parser.parse(sent)
    print('Real Result: ', result)
    print('My Result: ', tree)
    generate_graphviz(tree, 'parse_tree.gv')

('A', ('B', 'b'), ('C', ('A', 'a'), ('B', ('A', 'a'), ('C', 'c'))))

# test the pr evaluation
def test_eval_tree():
    t1 = ('A', ('B', 'b'), ('C', ('A', 'a'), ('B', ('A', 'a'), ('C', 'c'))))
    t2 = ('A', ('B', 'b'), ('C', ('B', 'b'), ('D', 'd')))
    t3 = ('A', ('B', 'b'), ('C', ('B', 'b')))
    print('Real: (0., 0.), Found: ', pr_eval(None, t1))
    print('Real: (1., 1.), Found: ', pr_eval(None, None))
    print('Real: (0.5, 0.333), Found: ', pr_eval(t1, t2))
    print('Real: (0.333, 1.), Found: ', pr_eval(t2, t1))
    print('Real: (1., 0.75), Found: ', pr_eval(t2, t3))
    print('Real: (1., 1.), Found: ', pr_eval(t1, t1))

# evaluate an existing example
def test_evaluate():
    parser = Syntax()
    parser.load_model('data/gram1.txt')
    parser.load_eval('data/test1.txt')
    parser.evaluate(-1)

def test_gen():
    parser = Syntax()
    parser.load_model('data/gram1.txt')
    parser.load_eval('data/test1.txt')
    for i in range(len(parser.eval)):
        if parser.eval[i][1] is not None:
            print(i)
            generate_graphviz(parser.eval[i][1], f'parse_tree{i}.gv')


if __name__ == '__main__':
    #test_cky()
    #test_eval_tree()
    #test_evaluate()
    test_gen()
