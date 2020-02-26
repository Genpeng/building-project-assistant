# _*_ coding: utf-8 _*_

"""
Bill Searcher.

Author: Genpeng Xu
"""

import faiss
import joblib
import nmslib
import numpy as np
import pandas as pd
from typing import List, Tuple, Union

# Own customized variables
from bpa.tokenizer import MyTokenizer
from bpa.global_variables import (BILL_DATA_FILEPATH,
                                  DATABASE_VECTORS_FILEPATH,
                                  T2_VECTORIZER_FILEPATH,
                                  ORDINAL_2_ID_DICT_FILEPATH,
                                  INDEX_TIME_PARAMS)


class NmslibBillSearcher(object):
    def __init__(self) -> None:
        self._db_df = pd.read_csv(BILL_DATA_FILEPATH)
        self._texts_df = self._generate_text_dataframe()
        self._tokenizer = MyTokenizer()
        self._vectorizer = joblib.load(T2_VECTORIZER_FILEPATH)
        self._ordinal_2_id = joblib.load(ORDINAL_2_ID_DICT_FILEPATH)
        self._generate_index()

    def _generate_index(self):
        vects = joblib.load(DATABASE_VECTORS_FILEPATH).toarray().astype('float32')
        self._index = nmslib.init(
            method="hnsw",
            space="l2",
            data_type=nmslib.DataType.DENSE_VECTOR
        )
        self._index.addDataPointBatch(vects)
        self._index.createIndex(INDEX_TIME_PARAMS)
        self._index.setQueryTimeParams({"efSearch": vects.shape[1]})

    def _generate_text_dataframe(self) -> pd.DataFrame:
        feature_cols = ['bill_name', 'bill_desc', 'unit']
        texts_df = self._db_df.copy()
        texts_df['bill_text'] = texts_df[feature_cols[0]].str.cat(
            texts_df[feature_cols[1:]], sep=' '
        )
        texts_df.drop(columns=feature_cols, inplace=True)
        return texts_df

    def find_k_nearest_bills_v1(self, query_texts: List[str], k: int = 5, num_threads: int = 4) -> List[pd.DataFrame]:
        text_segmented = [self._tokenizer.segment(text) for text in query_texts]
        query_vects = self._vectorizer.transform(text_segmented).toarray().astype('float32')
        nbrs = self._index.knnQueryBatch(query_vects, k, num_threads)
        results = []
        for i, text in enumerate(query_texts):
            ordinals, distances = nbrs[i]
            distances = list(distances)
            ids = set([self._ordinal_2_id[ordinal] for ordinal in ordinals])
            k_nearest_bills = self._db_df.loc[self._db_df.bill_id.isin(ids)]
            k_nearest_bills['distance'] = distances
            assert len(k_nearest_bills) == k
            results.append(k_nearest_bills)
        return results

    def find_k_nearest_bills_v2(self, query_texts: List[str], k: int = 5, num_threads: int = 4) -> List[pd.DataFrame]:
        text_segmented = [self._tokenizer.segment(text) for text in query_texts]
        query_vects = self._vectorizer.transform(text_segmented).toarray().astype('float32')
        nbrs = self._index.knnQueryBatch(query_vects, k, num_threads)
        results = []
        for i, text in enumerate(query_texts):
            ordinals, distances = nbrs[i]
            distances = list(distances)
            ids = [self._ordinal_2_id[ordinal] for ordinal in ordinals]
            k_nearest_bills = pd.DataFrame()
            if text in self._texts_df.bill_text.unique():
                bill_id = int(self._db_df.loc[self._texts_df.bill_text == text].bill_id)
                distances = [0] + distances
                k_nearest_bills = pd.concat([k_nearest_bills, self._db_df.loc[self._db_df.bill_id == bill_id]], axis=0)
            for _id in ids:
                k_nearest_bills = pd.concat([k_nearest_bills, self._db_df.loc[self._db_df.bill_id == _id]], axis=0)
            k_nearest_bills['distance'] = distances
            k_nearest_bills.drop_duplicates(['bill_name', 'bill_desc', 'unit'], keep='first', inplace=True)
            k_nearest_bills = k_nearest_bills.iloc[:k]
            assert len(k_nearest_bills) == k
            results.append(k_nearest_bills)
        return results


class FaissBillSearcher(object):
    def __init__(self) -> None:
        self._db_df = pd.read_csv(BILL_DATA_FILEPATH)
        self._texts_df = self._generate_text_dataframe()
        self._tokenizer = MyTokenizer()
        self._vectorizer = joblib.load(T2_VECTORIZER_FILEPATH)
        self._ordinal_2_id = joblib.load(ORDINAL_2_ID_DICT_FILEPATH)
        self._generate_index()

    def _generate_index(self):
        vects = joblib.load(DATABASE_VECTORS_FILEPATH).toarray().astype('float32')
        self._index = faiss.IndexFlatL2(vects.shape[1])
        self._index.add(vects)

    def _generate_text_dataframe(self) -> pd.DataFrame:
        feature_cols = ['bill_name', 'bill_desc', 'unit']
        texts_df = self._db_df.copy()
        texts_df['bill_text'] = texts_df[feature_cols[0]].str.cat(
            texts_df[feature_cols[1:]], sep=' '
        )
        texts_df.drop(columns=feature_cols, inplace=True)
        return texts_df

    def _find_k_nearest_indexes(self, query_texts: List[str], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        text_segmented = [self._tokenizer.segment(text) for text in query_texts]
        query_vects = self._vectorizer.transform(text_segmented).toarray().astype('float32')
        D, I = self._index.search(query_vects, k)
        return D, I

    def find_k_nearest_bills_v1(self, query_texts: List[str], k: int = 5) -> List[pd.DataFrame]:
        D, I = self._find_k_nearest_indexes(query_texts, k)
        results = []
        for i, text in enumerate(query_texts):
            ordinals, distances = I[i], list(D[i])
            ids = set([self._ordinal_2_id[ordinal] for ordinal in ordinals])
            res = self._db_df.loc[self._db_df.bill_id.isin(ids)]
            res['distance'] = distances
            assert len(res) == k
            results.append(res)
        return results

    def find_k_nearest_bills_v2(self, query_texts: List[str], k: int = 5) -> List[pd.DataFrame]:
        D, I = self._find_k_nearest_indexes(query_texts, k)
        results = []
        for i, text in enumerate(query_texts):
            ordinals, distances = I[i], list(D[i])
            ids = [self._ordinal_2_id[ordinal] for ordinal in ordinals]
            res = pd.DataFrame()
            if text in self._texts_df.bill_text.unique():
                bill_ids = set(self._texts_df.loc[self._texts_df.bill_text == text].bill_id.values)
                distances = [0] * len(bill_ids) + distances
                res = pd.concat([res, self._db_df.loc[self._db_df.bill_id.isin(bill_ids)]], axis=0)
            for _id in ids:
                res = pd.concat([res, self._db_df.loc[self._db_df.bill_id == _id]], axis=0)
            res["distance"] = distances
            res.drop_duplicates(['bill_name', 'bill_desc', 'unit'], keep='first', inplace=True)
            res = res.iloc[:k]
            assert len(res) == k
            results.append(res)
        return results


if __name__ == "__main__":
    query_texts = [
        "空心砖墙 1、砖品种、规格、强度等级：蒸压加气混凝土砌块 2、砂浆强度等级：M5水泥石灰砂浆 3、墙体厚度：200厚砖内墙 m3",  # index 0
        "直形墙 1、C40普通商品混凝土20石 m3",  # index 5
        "木质防火门 1、木质乙级防火门 2、门窗五金及油漆 3、详见门窗大样 m2",  # index 57
        "配电箱 1、 配电箱安装 AW1~7 台",  # index 68
    ]
    searcher1 = NmslibBillSearcher()
    results1 = searcher1.find_k_nearest_bills_v2(query_texts, 5)
    print(results1[0])

    # searcher2 = FaissBillSearcher()
    # results2 = searcher2.find_k_nearest_bills_v2(query_texts, 5)
    # print(results2[0])
