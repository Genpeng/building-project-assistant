# -*- coding: utf-8 -*-

"""
Test Web API of bill classification.

Author: Genpeng Xu (xgp1227atgmail.com)
"""

import os
import re
import json
import requests
import pandas as pd

SPEC_CHAR_REGEX = r"[ \"#%&\(\)\+,/:;<=>\?@\\\|]+"


def get_bill_classify_result(bill_id, bill_name, bill_desc, bill_unit):
    base_url = "http://47.115.112.147:9999/bill_classify?"
    params = "id=%s&name=%s&desc=%s&unit=%s" % (
        str(bill_id),
        re.sub(SPEC_CHAR_REGEX, "", bill_name),
        re.sub(SPEC_CHAR_REGEX, "", bill_desc),
        bill_unit,
    )
    return requests.get(base_url + params).json()


def get_bill_classify_result_batch(df):
    base_url = "http://47.115.112.147:9999/bill_classify?"
    bill_name = re.sub(SPEC_CHAR_REGEX, "", df.bill_name)
    bill_desc = re.sub(SPEC_CHAR_REGEX, "", df.bill_desc)
    params = "id=%s&name=%s&desc=%s&unit=%s" % (
        str(df.bill_id),
        bill_name,
        bill_desc,
        df.bill_unit,
    )
    return requests.get(base_url + params).json()


def get_k_nearest_bills(bill_id, bill_name, bill_desc, bill_unit, k=5):
    base_url = "http://47.115.112.147:9999/bill_similarity_search?"
    params = "id=%s&name=%s&desc=%s&unit=%s&k=%s" % (
        str(bill_id),
        re.sub(SPEC_CHAR_REGEX, "", bill_name),
        re.sub(SPEC_CHAR_REGEX, "", bill_desc),
        bill_unit,
        str(k),
    )
    return requests.get(base_url + params).json()


def get_k_nearest_bills_batch(df):
    base_url = "http://47.115.112.147:9999/bill_similarity_search?"
    bill_name = re.sub(SPEC_CHAR_REGEX, "", df.bill_name)
    bill_desc = re.sub(SPEC_CHAR_REGEX, "", df.bill_desc)
    params = "id=%s&name=%s&desc=%s&unit=%s&k=%s" % (
        str(df.bill_id),
        bill_name,
        bill_desc,
        df.bill_unit,
        str(df.k),
    )
    return requests.get(base_url + params).json()


def write_json_result_batch(df):
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filepath = os.path.join(output_dir, "%s.json" % str(df.bill_id))
    with open(output_filepath, "w", encoding="utf-8") as fout:
        json.dump(df.result, fout, ensure_ascii=False, indent=4)


def test_bill_classify():
    from pprint import pprint
    bill_id = "xxx123456"
    bill_name = "过梁模板"
    bill_desc = "1.类型：过@@^&()梁 2.其他++__做法：满足\\\\\\设计及现##行技@@术、质量验收规!!范@要求"
    bill_unit = "m2"
    pprint(
        get_bill_classify_result(bill_id, bill_name, bill_desc, bill_unit)
    )  # bill_type = "混凝土模板及支架(撑)"


def test_bill_classify_batch(input_fp):
    df = pd.read_excel(input_fp, sheet_name=0)
    df = df.iloc[:50]
    df["predicted_bill_type"] = df.apply(get_bill_classify_result_batch, axis=1)
    df["predicted_bill_type"] = df.predicted_bill_type.apply(lambda x: x["bill_type"])
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filepath = os.path.join(output_dir, "bill_classify_result.xlsx")
    df.to_excel(output_filepath, index=None)
    df["is_true"] = df.apply(lambda x: 1 if x.true_bill_type == x.predicted_bill_type else 0, axis=1)
    print("[INFO] The accuracy of bill classification is:", df.is_true.sum() / len(df))


def test_bill_similarity_search():
    from pprint import pprint
    bill_id = "xxx123456"
    bill_name = "直形墙"
    bill_desc = "1、C40普……@？？通商！！\\\\||||+品混$$$凝土20石"
    bill_unit = "m3"
    pprint(get_k_nearest_bills(bill_id, bill_name, bill_desc, bill_unit, 5))


def test_bill_similarity_search_batch(input_fp, k=5):
    df = pd.read_excel(input_fp, sheet_name=0)
    df["k"] = k
    ans = df[["bill_id"]]
    ans["result"] = df.apply(get_k_nearest_bills_batch, axis=1)
    ans.apply(write_json_result_batch, axis=1)


if __name__ == "__main__":
    # 【清单分类】单条测试
    # 测试方法：在下面函数的定义中修改相关变量
    test_bill_classify()

    # 【清单分类】批量测试
    # 测试方法：在下面的文件中添加需要测试的样例
    data_filepath = "data/bill-classify-test-samples.xlsx"
    test_bill_classify_batch(data_filepath)

    # 【清单相似搜索】单条测试
    # 测试方法：在下面函数的定义中修改相关变量
    test_bill_similarity_search()

    # 【清单相似搜索】批量测试
    # 测试方法：在下面的文件中添加需要测试的样例
    data_filepath = "data/bill-similarity-search-test-samples.xlsx"
    test_bill_similarity_search_batch(data_filepath, k=5)
