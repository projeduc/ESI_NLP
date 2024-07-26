import re
import os
import sys

from typing import Tuple, List

# Submit just this file, no zip, no additional files
# -------------------------------------------------

# Students:
#     - ...
#     - ...

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: What are the problem(s) with normalization in our case (Algerian tweets)?

A: ...

----------------------------------------------------------

----------------------------------------------------------
Q2: Why word similarity is based on edit distance and not vectorization such as TF?

A2: ...

----------------------------------------------------------

----------------------------------------------------------
Q3: Why tweets similarity is proposed as such? 
    (not another formula such as the sum of similarity of the first tweet's words with the second's divided by max length)

A3: ...

----------------------------------------------------------


----------------------------------------------------------
Q4: Why blanks are being duplicated before using regular expressions in our case?

A4: ...

----------------------------------------------------------

"""


# TODO Complete words similarity function
def word_sim(w1:str, w2:str) -> float:
    """Calculates Levenstein-based similarity between two words. 
    The function's words are interchangeable; i.e. levenstein(w1, w2) = levenstein(w2, w1)

    Args:
        w1 (str): First word.
        w2 (str): Second word.

    Returns:
        float: similarity.
    """

    if len(w1) * len(w2) == 0:
        return 0.0 # If one of them is empty then the distance is the length of the other

    D = []
    D.append([i for i in range(len(w2) + 1)])
    for i in range(len(w1)):
        l = [i+1]
        for j in range(len(w2)):
            s = D[i][j] + (0 if w1[i] == w2[j] else 1)
            m = min([s, D[i][j+1] + 1, l[j] + 1])
            l.append(m)
        D.append(l)
    
    return None


TASHKIIL	= [u'ِ', u'ُ', u'َ', u'ْ']
TANWIIN		= [u'ٍ', u'ٌ', u'ً']
OTHER       = [u'ـ', u'ّ']

# TODO Complete text normalization function
def normalize_text(text: str) -> str :
    """Normalize a text

    Args:
        text (str): source text

    Returns:
        str: result text
    """
    result = text.replace(' ', '  ') # duplicate the space
    result = re.sub('['+''.join(TASHKIIL+TANWIIN+OTHER)+']', '', result)
    result = result.lower()
    
    # SPCIAL 

    # FRENCH/ENGLISH/BERBER

    # DZ ARABIZI
    
    # ARABIC/DZ-ARABIC

    return re.sub(r'[./:,;?!؟…]', ' ', result)


#=============================================================================
#                         IMPLEMANTED FUNCTIONS
#=============================================================================

def get_similar_word(word:str, other_words:List[str]) -> Tuple[str, float]:
    """Get the most similar word with its similarity

    Args:
        word (str): a word
        other_words (List[str]): list of target words

    Returns:
        Tuple[str, float]: the most similar word from the target + its similarity 
    """

    mx_sim = 0.
    sim_word = ''
    for oword in other_words:
        sim = word_sim(word, oword)
        if sim > mx_sim:
            mx_sim = sim 
            sim_word = oword

    return sim_word, mx_sim


def tweet_sim(tweet1:List[str], tweet2:List[str]) -> float: 
    """Similarity between two tweets

    Args:
        tweet1 (List[str]): tokenized tweet 1
        tweet2 (List[str]): tokenized tweet 2

    Returns:
        float: their similarity
    """
    sim = 0.
    for word in tweet1:
        sim += get_similar_word(word, tweet2)[1]
    
    for word in tweet2:
        sim += get_similar_word(word, tweet1)[1] 

    return sim/(len(tweet1) + len(tweet2))


def get_tweets(url:str='DZtweets.txt') -> List[List[str]]:
    """Get tweets from a file, where each tweet is in a line

    Args:
        url (str, optional): the URL of tweets file. Defaults to 'DZtweets.txt'.

    Returns:
        List[List[str]]: A list of tokenized tweets
    """
    result = []
    with open(url, 'r', encoding='utf8') as f:
        for line in f:
            if len(line) > 1:
                line = normalize_text(line)
                tweet = line.split()
                result.append(tweet)
    return result


#=============================================================================
#                             TESTS
#=============================================================================

def _word_sim_test():
    tests = [
        ('amine', 'immature', 0.25),
        ('immature', 'amine', 0.25),
        ('', 'immature', 0.0),
        ('amine', '', 0.0),
        ('amine', 'amine', 1.0),
        ('amine', 'anine', 0.8),
        ('amine', 'anine', 0.8),
    ]
    
    for test in tests:
        sim = word_sim(test[0], test[1])
        print('-----------------------------------')
        print('similarity between ', test[0], ' and ', test[1])
        print('yours ', sim, ' must be ', test[2])


def _normalize_text_test():
    tests = [
        ('@adlenmeddi @faridalilatfr Est-il en vente a Alger?', 
         ['[USER]', '[USER]', 'est-il', 'en', 'vente', 'a', 'alger']),
        ('@Abderra51844745 @officialPACCI @AfcfT @UNDP Many thanks dear friend', 
         ['[USER]', '[USER]', '[USER]', '[USER]', 'many', 'than', 'dear', 'friend']),
        ('Info@shahanaquazi.com ; I love your profile.', 
         ['[MAIL]', 'i', 'love', 'your', 'profile']),
        ('âme à périt éclairées fète f.a.t.i.g.u.é.é', 
         ['ame', 'a', 'perit', 'eclairee', 'fete', 'f', 'a', 't', 'i', 'g', 'u', 'e', 'e']),
        ('palestiniens Manchester dangereuses dangereux écouter complètement vetements', 
         ['palestin', 'manchest', 'danger', 'danger', 'ecout', 'complet', 'vet']),
        ('reading followers naturally emotional traditions notably', 
         ['read', 'follow', 'natural', 'emotion', 'tradition', 'notab']),
        ('iggarzen Arnuyas', 
         ['iggarz', 'arnu']),
        ("it's That's don't doesn't", 
         ['it', 'is', 'that', 'is', 'do', 'not', 'does', 'not']),
        ("l'éventail s'abstenir qu'ont t'avoir j'ai D'or D'hier t'en l'aïd p'tit", 
         ['le', 'eventail', 'se', 'absten', 'que', 'ont', 'te', 'av', 'je', 'ai', 'de', 'or', 'de', 'hi', "t'", 'le', 'aïd', 'petit']),
        ('mal9itch mata3rfch Bsahtek ywaf9ek ya3tik 3ndk', 
         ['l9it', 'ta3rf', 'bsaht', 'ywaf9', 'ya3t', '3nd']),
        ('Khaltiha Khaltih yetfarjou fhamto mousiba wladi  Chawala khmouss', 
         ['khalti', 'khalti', 'yetfarj', 'fhamt', 'mousib', 'wlad', 'chawal', 'khmous']),
        ('لَا حـــــــــــــــــــــوْلَ وَلَا قُوَّةَ إِلَّا بِاللَّهِ الْعَزِيزُ الْحَكِيمُ،', 
         ['لا', 'حول', 'ولا', 'قوة', 'إلا', 'له', 'عزيز', 'حكيم،']),
        ('منلبسوش ميخرجش ميهمناش مايهمنيش قستيهاش فهمتش معليش', 
         ['ما', 'نلبس', 'ما', 'يخرج', 'ما', 'يهم', 'ما', 'يهم', 'ما', 'قستي', 'ما', 'فهمت', 'ما', 'عل']),
        ('الطاسيلي للاحباب اللهم المورال الاتحادبات المصلحين والتنازلات الجزائري فالناس للسونترال بروفايلات والصومال', 
         ['طاسيل', 'احباب', 'لهم', 'مور', 'اتحادب', 'مصلح', 'تنازل', 'جزائر', 'ناس', 'سونتر', 'بروفايل', 'صوم']),
        ('متشرفين نورمال تيميمون حلقات تركعوا عدوانية يفيقولو وعليكم بصيرته بصيرتها عملها عملهم', 
         ['متشرف', 'نورم', 'تيميم', 'حلق', 'تركع', 'عدواني', 'يفيقول', 'وعل', 'بصيرت', 'بصيرت', 'عمل', 'عمل']),
        ('رايحا طحتو توحشتك تبقاو ستوري راهي رميته الزنزانة وجيبوتي', 
         ['رايح', 'طحت', 'توحشت', 'تبقا', 'ستور', 'راه', 'رميت', 'زنزان', 'وجيبوت']),
    ]

    for test in tests:
        print('-----------------------------------')
        print('tweet ', test[0])
        print('your norm ', normalize_text(test[0]).split())
        print('must be', test[1])


def _tweet_sim_test():
    tweets = get_tweets() # If it cannot find the file, pass its URL as argument

    tests = [
        (1, 2, 0.45652173913043487),
        (4, 120, 0.40744680851063825),
        (5, 10, 0.3381987577639752),
        (204, 211, 0.4728021978021977),
        (15, 30, 0.48148148148148145),
        (50, 58, 0.3531746031746032),
        (100, 300, 0.5277777777777778),
    ]

    for test in tests:
        print('-----------------------------------')
        print('tweet 1', tweets[test[0]])
        print('tweet 2', tweets[test[1]])
        print('your sim ', tweet_sim(tweets[test[0]], tweets[test[1]]))
        print('must be  ', test[2])




# TODO activate one test at the time
if __name__ == '__main__':
    _word_sim_test()
    # _normalize_text_test()
    # _tweet_sim_test()

