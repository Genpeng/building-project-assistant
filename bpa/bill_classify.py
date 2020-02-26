# _*_ coding: utf-8 _*_

"""
Classify bill by using machine learning algorithms.

Author: Genpeng Xu
"""

from typing import List, Union

# Own customized modules
from bpa.classifier import BillClassifier

# global variables needed
clf = BillClassifier()


def classify_bill(texts: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(texts, str):
        texts = [texts]
    types = clf.classify_bill(texts)
    return types if len(types) > 1 else types[0]


if __name__ == '__main__':
    texts = [
        "矩形柱 1.混凝土种类:泵送 2.混凝土强度等级:C30 3.周长:1.6米以内 4.泵送高度:30-50米 5.部位:楼梯间柱 m3",
        "矩形柱（10F~14F）（楼梯柱） \"1.混凝土种类:预拌商品混凝土 2.混凝土强度等级:C30 3.:柱周长1.6m内，泵送高度超30m\"",
        "过梁模板 1.类型：过梁 2.其他做法：满足设计及现行技术、质量验收规范要求 m2",
    ]
    true_types = ['主体结构混凝土', '现浇混凝土柱', '混凝土模板及支架(撑)']
    types = classify_bill(texts)
    print(true_types)
    print(types)
