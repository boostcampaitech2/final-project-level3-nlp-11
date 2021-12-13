import requests
import time
import sys

from requests.api import request


class RequestBlog:
    def __init__(self) -> None:
        pass

    def get(self, url):
        try:
            response = requests.get(url, timeout=5)
        except:
            request_result = self.requset_again("get", url, None)
            if request_result:
                response = request_result
            else:
                print("error")
                sys.exit(1)

        return response

    def post(self, url, query):
        try:
            response = requests.post(url, json=query, timeout=5)
        except:
            requset_result = self.requset_again("post", url, query)
            if requset_result:
                response = requset_result
            elif not requset_result:
                print("error")
                sys.exit(1)

        return response

    def requset_again(self, def_name, url, query):
        if def_name == "post":
            time.sleep(0.1)
            return self.post(url, query)
        elif def_name == "get":
            time.sleep(0.1)
            return self.get(url)
        else:
            print("error")
            return None
