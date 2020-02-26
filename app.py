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
from bpa.bill_classify import classify_bill
from bpa.bill_similarity_search import find_k_nearest_bills

define("port", default=9999, help="server port", type=int)


class BillClassifyHandler(tornado.web.RequestHandler):
    def get(self):
        bill_id = self.get_argument("id")
        bill_name = self.get_argument("name")
        bill_desc = self.get_argument("desc")
        bill_unit = self.get_argument("unit")
        bill_type = classify_bill(bill_name + " " + bill_desc + " " + bill_unit)
        result = {
            "bill_id": bill_id,
            "bill_name": bill_name,
            "bill_desc": bill_desc,
            "bill_unit": bill_unit,
            "bill_type": bill_type,
        }
        self.write(json.dumps(result))


class BillSearchHandler(tornado.web.RequestHandler):
    def get(self):
        bill_id = self.get_argument("id")
        bill_name = self.get_argument("name")
        bill_desc = self.get_argument("desc")
        bill_unit = self.get_argument("unit")
        k = int(self.get_argument("k"))
        bill_text = bill_name + " " + bill_desc + " " + bill_unit
        k_nearest_bills = find_k_nearest_bills(bill_text, k)
        result = {
            "bill_id": bill_id,
            "bill_name": bill_name,
            "bill_desc": bill_desc,
            "bill_unit": bill_unit,
            "k_nearest_bills": k_nearest_bills.to_dict(orient="records"),
        }
        self.write(json.dumps(result))


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
