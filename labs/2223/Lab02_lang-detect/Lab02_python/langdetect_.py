#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: 
Mention one advantage of character-based language models over 
word-based ones in this task (language detection) and vice-versa.
----------------------------------------------------------


----------------------------------------------------------
Q2:
How can we enhance our n-gram model in order to better represent the probabilities?
In this case, which model will scale better (character-based or word-based) and why?
----------------------------------------------------------


----------------------------------------------------------
Q3:
During prediction, does the sentences' lengths affect the result? How? 
----------------------------------------------------------


----------------------------------------------------------
Q4:
We note that the accuracy of char-based model is 0.925 and that of word-based model is 0.125 
(all models have a recall and precision of 0 except Urdu). 
Why Char-based model gave better performance?
----------------------------------------------------------


"""

import math, json, os
from typing import Dict, List, Any, Tuple

# ====================================================
# ================= BiGram class =====================
# ====================================================

# TODO Complete Bigram class (fit, score, predict)
class BiGram:

    def __init__(self):
        self.uni_grams = {'<s>': 0, '</s>': 0}
        self.bi_grams = {}

    def fit(self, data:List[List[str]]) -> None:
        """Trains the current bi-gram model

        Args:
            data (List[List[str]]): a list of tokenized sentences.
        """
        pass

    # HINT: use math.log
    def score(self, past:str, current:str, alpha:float=1.) -> float:
        """Estimates the conditional probability P(current|past)

        Args:
            past (str): the past token.
            current (str): the current token.
            alpha (float, optional): Lidstone factor. Defaults to 1..

        Returns:
            float: conditional log-probability
        """
        pass

    def predict(self, tokens:List[str], alpha:float=1.) -> float:
        """Predicts the log probbability of a sequence of tokens
        P(t1 ... tn)

        Args:
            tokens (List[str]): a list of tokens (without padding).
            alpha (float, optional): Lidstone's factor. Defaults to 1..

        Returns:
            float: Log probability of the sequence.
        """
        pass

    # --------------- Implemented methods --------------
    def export_json(self) -> Dict[str, Any]:
        """Serialize the object as json object

        Returns:
            Dict[str, Any]: json representation of the current object.
        """
        return json.dumps(self.__dict__)

    def import_json(self, data:Dict[str, Any]):
        """Populate the current object using json serialization

        Args:
            data (Dict[str, Any]): json representation
        """
        for key in data:
            self.__dict__[key] = data[key]

# ====================================================
# ============== LangDetection class =================
# ====================================================
class LangDetector:
    def __init__(self, word=False):
        self.models:Dict[str, BiGram] = {}
        self.word:bool = word

    # private method to tokenize sentences
    def __tokenize(self, sent:str) -> List[str]:
        """Tokenize a sentence according to granularity (character vs word)

        Args:
            sent (str): the sentence.

        Returns:
            List[str]: a list of tokens.
        """
        if self.word:
            return sent.split()
        return list(sent)
         

    def fit(self, url:str):
        """Fit many models from some training data.
        The data is a list of files having the language code as name
        and ".train" as extension.

        Args:
            url (str): the URL of the folder containing ".train" files.
        """
        for fp in os.listdir(url):
            if fp.endswith(".train"):
                code = os.path.splitext(fp)[0]
                fp = os.path.join(url, fp)
                data = []
                f = open(fp, 'r')
                for l in f: #Browse line by line
                    l = l.strip(' \t\r\n')
                    if len(l) > 0:
                         data.append(self.__tokenize(l))
                f.close()
                self.models[code] = BiGram()
                self.models[code].fit(data)

    def predict(self, sent:str, alpha:float=1.) -> str:
        """Predicts the language of a given sentence. 


        Args:
            sent (str): A sentence.
            alpha (float, optional): Lidstone's factor. Defaults to 1..

        Returns:
            str: Language's code.
        """
        tokens = self.__tokenize(sent)
        p_max, c_max = -math.inf, ''
        for c in self.models:
            p = self.models[c].predict(tokens, alpha=alpha)
            if p > p_max:
                p_max, c_max = p, c
        return c_max


    def populate_model(self, url:str):
        """Fill the model from a json serialized object

        Args:
            url (str): URL of json file
        """
        f = open(url, 'r')
        data = json.load(f)
        self.word = data['word']
        self.models = {}
        for code in data['models']:
            self.models[code] = BiGram()
            self.models[code].import_json(data['models'][code])
        f.close()

    def save_model(self, url:str):
        """Serialize the model into a json file

        Args:
            url (str): URL of the json file.
        """
        f = open(url, 'w')
        json.dump(self.__dict__, f, default=lambda o: o.export_json())
        f.close()

    # private static method to load evaluation samples
    def __get_evaluation(self, url:str) -> List[Tuple[str, str]]:
        result = []
        f = open(url, 'r')
        for l in f:
            l = l.strip(' \t\r\n')
            if len(l) < 2:
                continue
            info = l.split('#')
            result.append((info[0], info[1]))
        f.close()
        return result

    def evaluate(self, url:str, alpha:float=1.) -> Dict[str, Any]:
        total = 0
        found = 0
        counts = {}# code: [true, pred, true.pred]
        for sent, code in self.__get_evaluation(url):
            pred = self.predict(sent, alpha=alpha)
            if code not in counts:
                counts[code] = [0, 0, 0]
            if pred not in counts:
                counts[pred] = [0, 0, 0]
            counts[code][0] += 1
            counts[pred][1] += 1
            if pred == code:
                found += 1
                counts[code][2] += 1
            total += 1
        
        res = {'accuracy': found/total}
        for code in counts:
            res[code] = {
                'R': 0.0 if counts[code][0] == 0 else counts[code][2]/counts[code][0],
                'P': 0.0 if counts[code][1] == 0 else counts[code][2]/counts[code][1],
            }
            
        return res


test = [
        ['a', 'b', 'a', 'a'],
        ['b', 'c', 'a'],
        ['a', 'a']
    ]

# ====================================================
# ===================== Tests ========================
# ====================================================
def test_bigram_fit():
    bigram = BiGram()
    bigram.fit(test)
    print('========= BiGram fit test =========')
    print('your unigrams', bigram.uni_grams)
    print('must be', {'<s>': 3, '</s>': 3, 'a': 6, 'b': 2, 'c': 1})
    print('your bigrams', bigram.bi_grams)
    print('must be', {'<s>#a': 2, 'a#b': 1, 'b#a': 1, 'a#a': 2, 
                      'a#</s>': 3, '<s>#b': 1, 'b#c': 1, 'c#a': 1})

def test_bigram_score():
    bigram = BiGram()
    bigram.fit(test)
    print('========= BiGram score test =========')
    print('p_1.0(a|b)=-1.252762968495368, found', bigram.score('b', 'a', alpha=1.))
    print('p_0.5(a|b)=-1.098612288668109, found', bigram.score('b', 'a', alpha=.5))
    print('p_1.0(b|b)=-1.945910149055313, found', bigram.score('b', 'b', alpha=1.))
    print('p_0.5(b|b)=-2.197224577336219, found', bigram.score('b', 'b', alpha=.5))

    print('p_1.0(a|d)=-1.609437912434100, found', bigram.score('d', 'a', alpha=1.))
    print('p_0.5(a|d)=-1.609437912434100, found', bigram.score('d', 'a', alpha=.5))

def test_bigram_predict():
    bigram = BiGram()
    bigram.fit(test)
    print('========= BiGram predict test =========')
    print('p_1.0(aba)=-4.949941225423999, found', bigram.predict(['a', 'b', 'a'], alpha=1.))
    print('p_0.5(aba)=-4.633271616098966, found', bigram.predict(['a', 'b', 'a'], alpha=.5))

    print('p_1.0(ada)=-5.999763349922677, found', bigram.predict(['a', 'd', 'a'], alpha=1.))
    print('p_0.5(ada)=-6.242709528533067, found', bigram.predict(['a', 'd', 'a'], alpha=.5))


def test_landetect():
    program_char = LangDetector()
    program_word = LangDetector(word=True)

    program_char.fit('data')
    program_word.fit('data')

    # program_char.save_model('./char_based.json')
    # program_word.save_model('./word_based.json')

    print('------- char ----------')
    print(program_char.evaluate('data/lang.eval'))
    print('------- word ----------')
    print(program_word.evaluate('data/lang.eval'))


# ====================================================
# =================== MAIN PROGRAM ===================
# ====================================================

# Activate unitary tests one by one and desactivate the rest.
if __name__ == '__main__':
    test_bigram_fit()
    # test_bigram_score()
    # test_bigram_predict()
    # test_landetect()

