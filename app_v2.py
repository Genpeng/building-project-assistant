# -*- coding: utf-8 -*-

"""
The Web APIs based on tornado

Author: Genpeng Xu (xgp1227atgmail.com)
"""

import json
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.options import define, options
from pandas.io.json import json_normalize
from bpa.bill_classify import classify_bill
from bpa.bill_similarity_search import find_k_nearest_bills

define("port", default=8888, help="server port", type=int)


class BillClassifyHandler(tornado.web.RequestHandler):
    def post(self):
        json_data = json.loads(self.get_argument("data"))
        df = json_normalize(json_data)
        if "unit" in set(df.columns):
            texts = list((df.name + " " + df.attr + " " + df.unit).values)
        else:
            texts = list((df.name + " " + df.attr).values)
        df["predicted_bill_type"] = classify_bill(texts)
        data_list = df.to_dict(orient="records")
        return self.write(json.dumps(data_list, ensure_ascii=False, indent=2))


class BillSearchHandler(tornado.web.RequestHandler):
    def post(self):
        k = int(self.get_argument("k"))
        json_data = json.loads(self.get_argument("data"))
        df = json_normalize(json_data)
        if "unit" in set(df.columns):
            texts = list((df.name + " " + df.attr + " " + df.unit).values)
        else:
            texts = list((df.name + " " + df.attr).values)
        dfs_list = find_k_nearest_bills(texts, k)
        if isinstance(dfs_list, list):
            json_list = [df.to_dict(orient="records") for df in dfs_list]
            df["k_nearest_bills"] = json_list
        else:
            df["k_nearest_bills"] = [dfs_list.to_dict(orient="records")]
        data_list = df.to_dict(orient="records")
        return self.write(json.dumps(data_list, ensure_ascii=False, indent=2))


def make_app():
    return tornado.web.Application(
        [
            (r"/bill_classify", BillClassifyHandler),
            (r"/bill_similarity_search", BillSearchHandler),
        ]
    )


if __name__ == "__main__":
    options.parse_command_line()
    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
