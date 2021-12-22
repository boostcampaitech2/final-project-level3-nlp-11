import os
import json
import requests

from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class PostSurvey:
    def __init__(self) -> None:
        pass

    def post_servey(self, query, area, place, is_good):
        base_url = os.getenv("API_BASE_URL")
        url = f"{base_url}survey"
        data = {
            "query": query,
            "location": area,
            "place": place,
            "is_good": is_good,
        }
        res = requests.post(url, data=json.dumps(data))
