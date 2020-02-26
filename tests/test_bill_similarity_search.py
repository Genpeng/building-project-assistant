# _*_ coding: utf-8 _*_

from bpa.bill_similarity_search import find_k_nearest_bills


def load_test_data(n):
    import numpy as np
    import pandas as pd
    np.random.seed(89)
    df = pd.read_csv("./test_data/test_data_for_similarity_search.txt")
    random_indexes = np.random.permutation(len(df))[-n:]
    return list(df.iloc[random_indexes, :]['bill_text'].values)


def test_bill_similarity_search_performance():
    from time import time
    n = 5000
    query_texts = load_test_data(n)
    k = 5
    t0 = time()
    ans = find_k_nearest_bills(query_texts, k)
    print(ans[0])
    print("[INFO] Find %d bill's top %d nearest bills took %f seconds." % (n, k, time() - t0))
