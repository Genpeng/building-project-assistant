# -*- coding: utf-8 -*-

"""
Test Web API of bill classification.

Author: Genpeng Xu (xgp1227atgmail.com)
"""

import os
import json
import requests
from urllib.parse import quote
from pandas.io.json import json_normalize

OUTPUT_DIR = "./output/"


def get_bill_classify_result(data):
    url = "http://47.115.112.147:8888/bill_classify?data=%s" % data
    return requests.post(url).json()


def get_k_nearest_bills(data, k):
    url = "http://47.115.112.147:8888/bill_similarity_search?k=%s&data=%s" % (str(k), data)
    return requests.post(url).json()


def test_bill_classify(data_filepath):
    with open(data_filepath, "r", encoding="utf-8") as fin:
        json_data = json.load(fin)
    data_str = quote(json.dumps(json_data, ensure_ascii=False))
    ans_json = get_bill_classify_result(data_str)
    df_ans = json_normalize(ans_json)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    output_filepath = os.path.join(OUTPUT_DIR, "bill-classify-result.xlsx")
    df_ans.to_excel(output_filepath, index=None)
    df_ans["is_true"] = df_ans.apply(
        lambda x: 1 if x.subjectModelName == x.predicted_bill_type else 0,
        axis=1
    )
    print("[INFO] The accuracy of bill classification is: %.2f" % (df_ans.is_true.sum() / len(df_ans) * 100))
    print("[INFO] The result is stored in the `output` directory!")


def test_bill_similarity_search(data_filepath, k):
    with open(data_filepath, "r", encoding="utf-8") as fin:
        json_data = json.load(fin)
    data_str = quote(json.dumps(json_data, ensure_ascii=False))
    ans_json = get_k_nearest_bills(data_str, k)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    output_filepath = os.path.join(OUTPUT_DIR, "bill-similarity-search-result.json")
    with open(output_filepath, "w", encoding="utf-8") as fout:
        json.dump(ans_json, fout, ensure_ascii=False, indent=2)
    print("[INFO] Finished to find k nearest bills! ( ^ _ ^ ) V")
    print("[INFO] The result JSON is stored in the `output` directory!")


if __name__ == "__main__":
    # 【清单分类测试】
    # 测试方法：指定需要测试的json数据路径，返回结果在 `output` 目录下
    data_filepath = "data/bill-classify/bill_data_random_100.json"
    test_bill_classify(data_filepath)

    # 【清单相似度测试】
    # 测试方法：指定需要测试的json数据路径，返回结果在 `output` 目录下
    data_filepath = "data/bill-similarity-search/standard_bill_random_10.json"
    k = 5
    test_bill_similarity_search(data_filepath, k)
