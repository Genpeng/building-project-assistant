# _*_ coding: utf-8 _*_

"""
Bill Classifier.

Author: Genpeng Xu
"""

import joblib
from typing import List

# Own customized variables & modules
from bpa.tokenizer import MyTokenizer
from bpa.global_variables import (T1_VECTORIZER_FILEPATH,
                                  T1_MODEL_FILEPATH,
                                  LABEL_2_TYPE_DICT_FILEPATH)


class BillClassifier(object):
    def __init__(self):
        self._tokenizer = MyTokenizer()
        self._vectorizer = joblib.load(T1_VECTORIZER_FILEPATH)
        self._model = joblib.load(T1_MODEL_FILEPATH)
        self._label_2_type = joblib.load(LABEL_2_TYPE_DICT_FILEPATH)

    def _classify(self, texts: List[str]) -> List[int]:
        texts_segmented = [self._tokenizer.segment(text) for text in texts]
        return list(self._model.predict(self._vectorizer.transform(texts_segmented)))

    def classify_bill(self, texts: List[str]) -> List[str]:
        labels = self._classify(texts)
        return [self._label_2_type[label] for label in labels]


if __name__ == "__main__":
    texts = [
        "矩形柱 1.混凝土种类:泵送 2.混凝土强度等级:C30 3.周长:1.6米以内 4.泵送高度:30-50米 5.部位:楼梯间柱 m3",
        "矩形柱（10F~14F）（楼梯柱） \"1.混凝土种类:预拌商品混凝土 2.混凝土强度等级:C30 3.:柱周长1.6m内，泵送高度超30m\"",
        "过梁模板 1.类型：过梁 2.其他做法：满足设计及现行技术、质量验收规范要求 m2",
    ]
    true_types = ['主体结构混凝土', '现浇混凝土柱', '混凝土模板及支架(撑)']
    clf = BillClassifier()
    print(true_types)
    print(clf.classify_bill(texts))
