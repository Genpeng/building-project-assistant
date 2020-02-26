# _*_ coding: utf-8 _*_

"""
Customized tokenizer.

Author: Genpeng Xu
"""

import re
import jieba
import string
from zhon import hanzi

# Own customized modules
from bpa.util import load_stopwords
from bpa.global_variables import (USERDICT_FILEPATH,
                                  STOPWORDS_FILEPATH)


class MyTokenizer(object):
    PUNC_REGEX = r"^[{} \s]+$".format(string.punctuation + hanzi.punctuation)
    NUM_REGEX = r"^[0-9]*\.?[0-9]+$"
    UNIT_REGEX = r"^([0-9]*)(mm|cm|m)?$"
    ALPHANUM_REGEX = r"^[a-zA-Z0-9]+$"

    def __init__(self) -> None:
        jieba.load_userdict(USERDICT_FILEPATH)
        self._stopwords = load_stopwords(STOPWORDS_FILEPATH)

    def segment(self, text: str, lowercase: bool = True) -> str:
        if lowercase:
            text = text.lower()
        words = []
        jieba_res = jieba.cut(text)
        for w in jieba_res:
            if len(w) <= 1:
                continue
            if w in self._stopwords:
                continue
            if re.match(MyTokenizer.PUNC_REGEX, w):
                continue
            if re.match(MyTokenizer.NUM_REGEX, w):
                continue
            if re.match(MyTokenizer.UNIT_REGEX, w):
                continue
            words.append(w)
        return ' '.join(words)


if __name__ == "__main__":
    text = "空心砖墙,1、砖品种、规格、强度等级：蒸压加气混凝土砌块 2、砂浆强度等级：M5水泥石灰砂浆 3、墙体厚度：200厚砖内墙,m3"
    tokenizer = MyTokenizer()
    print(tokenizer.segment(text))
