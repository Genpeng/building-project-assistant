# -*- coding: utf-8 -*-

"""
Test Web API of bill classification.

Author: Genpeng Xu (xgp1227atgmail.com)
"""

import requests


def get_bill_classify_result(bill_id, bill_name, bill_desc, bill_unit):
    base_url = "http://47.115.112.147:9999/bill_classify?"
    params = "id=%s&name=%s&desc=%s&unit=%s" % (bill_id, bill_name, bill_desc, bill_unit)
    return requests.get(base_url + params).json()


def get_k_nearest_bills(bill_id, bill_name, bill_desc, bill_unit, k=5):
    base_url = "http://47.115.112.147:9999/bill_similarity_search?"
    params = "id=%s&name=%s&desc=%s&unit=%s&k=%s" % (bill_id, bill_name, bill_desc, bill_unit, k)
    return requests.get(base_url + params).json()


def test_bill_classify():
    from pprint import pprint
    bill_id = "xxx123456"
    bill_name = "过梁模板"
    bill_desc = "1.类型：过梁 2.其他做法：满足设计及现行技术、质量验收规范要求"
    bill_unit = "m2"
    pprint(get_bill_classify_result(bill_id, bill_name, bill_desc, bill_unit))  # bill_type = "混凝土模板及支架(撑)"


def test_bill_similarity_search():
    from pprint import pprint
    bill_id = "5"
    bill_name = "直形墙"
    bill_desc = "1、C40普通商品混凝土20石"
    bill_unit = "m3"
    pprint(get_k_nearest_bills(bill_id, bill_name, bill_desc, bill_unit, 5))


if __name__ == "__main__":
    test_bill_classify()
    test_bill_similarity_search()
