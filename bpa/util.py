# _*_ coding: utf-8 _*_

"""
Some utility functions.

Author: Genpeng Xu
"""

import numpy as np
import pandas as pd
from typing import Tuple, Set


def split_data_set(df: pd.DataFrame,
                   label_col: str,
                   test_percent: float = 0.2,
                   min_test_num: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    labels = set(df[label_col])
    for label in labels:
        temp_df = df.loc[df[label_col] == label]
        n = len(temp_df)
        n_test = int(n * test_percent)
        if n_test < min_test_num:
            continue
        else:
            random_indexes = np.random.permutation(n)
            train_indexes = random_indexes[:-n_test]
            test_indexes = random_indexes[-n_test:]
            df_train = pd.concat([df_train, temp_df.iloc[train_indexes]], axis=0)
            df_test = pd.concat([df_test, temp_df.iloc[test_indexes]], axis=0)
    return df_train, df_test


def load_stopwords(filepath: str) -> Set:
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        stopwords.add(line.strip())
    return stopwords
