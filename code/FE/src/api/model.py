import os
import json
import requests

from api.tour_api import Api
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class ModelApi:
    def __init__(self) -> None:
        pass

    def request_predict(self, query, area):
        # http://49.50.163.245:6011/
        base_url = os.getenv("API_BASE_URL")
        url = f"{base_url}predict/"
        params = {"query": query, "location": area}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            result = response.json()
            return Api().get_tour_topK(result["places"], result["location"])
        else:
            return None

    def request_similar(self, query, area):
        base_url = os.getenv("API_BASE_URL")
        url = f"{base_url}get_similar/"
        params = {"query": query, "location": area}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            result = response.json()
            return Api().get_tour_topK(result["places"], result["location"])
        else:
            return None
