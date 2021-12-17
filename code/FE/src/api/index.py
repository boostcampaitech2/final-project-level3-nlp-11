import requests
import json
import numpy as np
import urllib.parse
import os
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class Api:
    def get_tour(self, spot, location):
        data_path = os.getenv("DATA_PATH")
        key = os.getenv("TOUR_API_TOKEN_KEY")
        key = urllib.parse.unquote(key)
        with open(
            f"{data_path}Sparse/tour_spot_name_without_cultural_contentid.json"
        ) as f:
            reload = json.load(f)

        for area in reload:
            if spot in reload[area]["관광지"]:
                contentid = reload[area]["관광지"][spot]["contentid"]

        url = "http://api.visitkorea.or.kr/openapi/service/rest/KorService/detailCommon"
        param = {
            "ServiceKey": key,
            "MobileOS": "ETC",
            "MobileApp": "AppTest",
            "_type": "json",
            "defaultYN": "Y",
            "addrinfoYN": "Y",
            "overviewYN": "Y",
            "firstImageYN": "Y",
            "firstImage2YN": "Y",
            "contentId": contentid,  # 장소를 특정
        }
        context_images = requests.get(url, params=param)
        return context_images.json()

    def get_tour_topK(self, input, location):
        result = []
        for name in input:
            info = {
                "spot": name,
                "location": location,
            }
            response = self.get_tour(name, location)
            if not "item" in response["response"]["body"]["items"]:
                continue
            response = response["response"]["body"]["items"]["item"]

            if "firstimage" in response:
                info["img_url"] = response["firstimage"]
            else:
                info["img_url"] = "https://missioninfra.net/img/noimg/noimg_4x3.gif"
            if "addr1" in response:
                info["addr1"] = response["addr1"]
            else:
                info["addr1"] = "주소 정보 없음"

            if "tel" in response:
                info["tel"] = response["tel"]
            else:
                info["tel"] = "전화 정보 없음"
            info["context"] = response["overview"]
            result.append(info)

        return result
