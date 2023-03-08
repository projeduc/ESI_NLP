import re
import os
import sys

from typing import Tuple, List

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: 
We noticed that some words are not taken in consideration although they have more chance.
For example, the word "qsministrator" against "administrator". 
Why? Propose an amelioration.
----------------------------------------------------------


----------------------------------------------------------
Q2:
Some words are not detected as correct although they are.
Example, the word "confirmation" is correct, but detected as a spell error.
The dictionary contains only the word "confirm".
Propose a solution without changing the content of the dictionary.
----------------------------------------------------------


----------------------------------------------------------
Q3:
We want to evaluate our system. 
For each word, we prepared a list of possible corrections ordered by order of similarity.
Propose a method to evaluate such system.
----------------------------------------------------------

"""


# TODO Complete levenstein function
def levenstein(w1:str, w2:str, sub:int=2) -> Tuple[int, List[List[int]]]:
    """Calculates Levenstein distance between two words. 
    The function's words are interchangeable; i.e. levenstein(w1, w2) = levenstein(w2, w1)

    Args:
        w1 (str): First word.
        w2 (str): Second word.
        sub (int, optional): Substitution's cost. Defaults to 2.

    Returns:
        Tuple[int, List[List[int]]]: distance + dynamic table.
    """

    D = []

    return D[-1][-1], D

# TODO Complete Choosing similar words function
def choose_words(word:str, choices:List[str], th:float=0.5, sub:int=2) -> Tuple[List[str], List[int]]:
    """Choose similar words to a given one from a list 

    Args:
        word (str): The word
        choices (List[str]): A list from which similar words are drawn (chosen).
        th (float, optional): Threshold of similarity. Defaults to 0.5.
        sub (int, optional): Substitution's cost. Defaults to 2.

    Returns:
        Tuple[List[str], List[int]]: List of similar words and list of distances. 
    """

    return [], []


def get_words_list(word:str, dict_url:str) -> List[str]:
    """Getting a list of words from a given dictionary. 
    These words must have the same first letter or second letter of
    a given word

    Args:
        word (str): The word to which we compare
        dict_url (str): the dictionary's link

    Returns:
        List[str]: List of similar words
    """
    # The lines containing words which can be similar to the word in question
    lines = set()
    # We search in the first index for lines with words 
    # having the same first letter as the word in question
    with open(dict_url[:-4] + 'idx1', 'r', encoding='utf8') as f:
        for line in f:
            if len(line) > 0 and line[0] == word[0]:
                lines.update(line[2:].split(','))
                break

    # We search in the second index for lines with words 
    # having the same second letter as the word in question
    with open(dict_url[:-4] + 'idx2', 'r', encoding='utf8') as f:
        for line in f:
            if len(line) > 0 and line[0] == word[1]:
                lines.update(line[2:].split(','))
                break
    
    
    # If there is no word in the dictionary with the same first letter of second
    # Then no need to open the dictionary
    if len(lines) == 0:
        return [] 
    
    # Get all words by their position (line) in dictionary file
    result = []
    with open(dict_url, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if str(i) in lines:
                result.append(line[:-1]) # The last charcater is line break \n
    return result


def check_word(word:str, dict_url:str='data/eng.dict') -> List[str]:
    """Check a word's spelling

    Args:
        word (str): the word in question.
        dict_url (str, optional): the dictionary's path. Defaults to 'data/eng.dict'.

    Returns:
        List[str]: a list containing the word itself if no error exists. 
                   Otherwise, some corrections.
    """
    possible_words = get_words_list(word, dict_url=dict_url)
    if word in possible_words:
        return [word]
    
    words, scores =  choose_words(word, possible_words, th=0.75, sub=1)
    return words


#=============================================================================
#                             TESTS
#=============================================================================

def _levenstein_test():
    tests = [
        ('amine', 'immature', 2, 9, [[0, 1, 2, 3, 4, 5, 6, 7, 8], 
                                     [1, 2, 3, 4, 3, 4, 5, 6, 7], 
                                     [2, 3, 2, 3, 4, 5, 6, 7, 8], 
                                     [3, 2, 3, 4, 5, 6, 7, 8, 9], 
                                     [4, 3, 4, 5, 6, 7, 8, 9, 10], 
                                     [5, 4, 5, 6, 7, 8, 9, 10, 9]]),
        ('immature', 'amine', 2, 9, [[0, 1, 2, 3, 4, 5], 
                                     [1, 2, 3, 2, 3, 4], 
                                     [2, 3, 2, 3, 4, 5], 
                                     [3, 4, 3, 4, 5, 6], 
                                     [4, 3, 4, 5, 6, 7], 
                                     [5, 4, 5, 6, 7, 8], 
                                     [6, 5, 6, 7, 8, 9], 
                                     [7, 6, 7, 8, 9, 10], 
                                     [8, 7, 8, 9, 10, 9]]),
        ('', 'immature', 2, 8, []),
        ('amine', '', 2, 5, []),
        ('amine', 'amine', 2, 0, [[0, 1, 2, 3, 4, 5], 
                                  [1, 0, 1, 2, 3, 4], 
                                  [2, 1, 0, 1, 2, 3], 
                                  [3, 2, 1, 0, 1, 2], 
                                  [4, 3, 2, 1, 0, 1], 
                                  [5, 4, 3, 2, 1, 0]]),
        ('amine', 'anine', 2, 2, [[0, 1, 2, 3, 4, 5], 
                                  [1, 0, 1, 2, 3, 4], 
                                  [2, 1, 2, 3, 4, 5], 
                                  [3, 2, 3, 2, 3, 4], 
                                  [4, 3, 2, 3, 2, 3], 
                                  [5, 4, 3, 4, 3, 2]]),
        ('amine', 'anine', 1, 1, [[0, 1, 2, 3, 4, 5], 
                                  [1, 0, 1, 2, 3, 4], 
                                  [2, 1, 1, 2, 3, 4], 
                                  [3, 2, 2, 1, 2, 3], 
                                  [4, 3, 2, 2, 1, 2], 
                                  [5, 4, 3, 3, 2, 1]]),
    ]
    
    for test in tests:
        d, D = levenstein(test[0], test[1], sub=test[2])
        print('-----------------------------------')
        print('levenstein test for ', test[0], ' and ', test[1])
        print('your distance ', d, ' must be ', test[3])
        print('your table ', D, ' must be ', test[4])

def _choose_words_test():
    tests = [
        ('tsk', ['task', 'ask', 'asking', 'take'], 0.25, (['task', 'ask', 'take'], [1, 2, 3])), 
        ('tsk', ['task', 'ask', 'asking', 'take'], 0.75, (['task'], [1])), 
    ]

    for test in tests:
        print('-----------------------------------')
        print('word:', test[0], ', choices:', test[1], ' threshold:', test[2])
        print('Your selection:', choose_words(test[0], test[1], th=test[2]),
              ' must be', test[3])
    


if __name__ == '__main__':
    _levenstein_test()
    # _choose_words_test()
    # print(get_words_list('immature', 'data/eng.dict'))
    # print('word=tsk', 'corrections:', check_word('tsk))

