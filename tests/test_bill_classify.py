# _*_ coding: utf-8 _*_

from typing import List, Tuple
from bpa.bill_classify import classify_bill


def load_test_samples(filepath: str) -> Tuple[List[str], List[str]]:
    import pandas as pd
    df = pd.read_csv(filepath)
    texts = list(df['text'])
    types = list(df['type'])
    return texts, types


def test_classify():
    texts, types = load_test_samples("./test_data/test_data_for_acc.txt")
    preds = classify_bill(texts)
    assert preds == types


def test_performance():
    import time
    texts, labels = load_test_samples("./test_data/test_data_for_performance.txt")
    # texts, labels = texts[-1:], labels[-1:]
    t0 = time.time()
    classify_bill(texts)
    print("[INFO] Classify %d documents took %f seconds." % (len(texts), time.time() - t0))


if __name__ == '__main__':
    test_classify()
    test_performance()
