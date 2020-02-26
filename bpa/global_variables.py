# _*_ coding: utf-8 _*_

"""
Some useful global variables.

Author: Genpeng Xu
"""

import os

ROOT_DIR = os.path.dirname(__file__)

DATA_DIR = os.path.join(ROOT_DIR, 'db')
BILL_DATA_FILEPATH = os.path.join(DATA_DIR, 'standard-bills.txt')

DICT_DIR = os.path.join(ROOT_DIR, "dicts")
USERDICT_FILEPATH = os.path.join(DICT_DIR, 'userdict.txt')
STOPWORDS_FILEPATH = os.path.join(DICT_DIR, 'stopwords.txt')

# things about task 1 (classify)
T1_DATA_DIR = os.path.join(ROOT_DIR, 't1_model')
LABEL_2_TYPE_DICT_FILEPATH = os.path.join(T1_DATA_DIR, 'label_2_type.dict')
TYPE_2_LABEL_DICT_FILEPATH = os.path.join(T1_DATA_DIR, 'type_2_label.dict')
T1_VECTORIZER_FILEPATH = os.path.join(T1_DATA_DIR, 'vectorizer.joblib')
T1_MODEL_FILEPATH = os.path.join(T1_DATA_DIR, 'model.joblib')

# things about task 2 (similarity search)
T2_DATA_DIR = os.path.join(ROOT_DIR, 't2_model')
DATABASE_VECTORS_FILEPATH = os.path.join(T2_DATA_DIR, 'database_vectors.joblib')
ORDINAL_2_ID_DICT_FILEPATH = os.path.join(T2_DATA_DIR, 'ordinal_2_id.dict')
T2_VECTORIZER_FILEPATH = os.path.join(T2_DATA_DIR, 'vectorizer.joblib')

# nmslib bill searcher configuration
INDEX_TIME_PARAMS = {
    "M": 15,
    "indexThreadQty": 4,
    "efConstruction": 100,
}
QUERY_TIME_PARAMS = {
    "efSearch": 7000
}

if __name__ == '__main__':
    print(ROOT_DIR)

    print(DATA_DIR)
    print(BILL_DATA_FILEPATH)

    print(DICT_DIR)
    print(USERDICT_FILEPATH)
    print(STOPWORDS_FILEPATH)

    print(T1_DATA_DIR)
    print(LABEL_2_TYPE_DICT_FILEPATH)
    print(TYPE_2_LABEL_DICT_FILEPATH)
    print(T1_VECTORIZER_FILEPATH)
    print(T1_MODEL_FILEPATH)

    print(T2_DATA_DIR)
    print(DATABASE_VECTORS_FILEPATH)
    print(ORDINAL_2_ID_DICT_FILEPATH)
    print(T2_VECTORIZER_FILEPATH)
