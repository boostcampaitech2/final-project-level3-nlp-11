import os
import urllib.parse
import json
import time
import re
from bs4 import BeautifulSoup
from utils.request_blog import RequestBlog
from pathlib import Path
from utils.replace_text import ReplaceText
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class Crawling_naver:
    def __init__(self, query) -> None:
        self.query = query

    def get_result(self):
        query = self.query
        id, name = self.get_id(query)
        info = {"query": query, "search_name": name, "total": 0, "Error": []}
        if id == -1:
            return None, info
        urls = self.get_blog_url(id)
        if not urls:
            print(f"{query}:blog 정보 없음")
            return None, info
        contexts = []
        start = time.time()
        for i, url in enumerate(urls):
            now = time.time()
            print(f"\r{query} {i+1} runtime: {now - start:.2f}", end="")
            text, url = self.get_context(url)
            if not text:
                info["Error"].append(url)
                continue
            contexts.append({"context": text, "url": url})

        # contexts = list(filter(None,contexts))
        info["total"] = len(contexts)
        print("\n")
        return contexts, info

    def get_id(self, query):

        query_ = urllib.parse.quote_plus(query)

        url = f"https://m.map.naver.com/search2/searchMore.naver?query={query_}&sm=clk&style=v5&page=1&displayCount=75&type=SITE_1"

        response = json.loads(RequestBlog().get(url).text)

        if not "site" in response["result"]:
            print(f'query:{query} "정보 없음"')
            id = -1
            return id, None
        id = response["result"]["site"]["list"][0]["id"]
        name = response["result"]["site"]["list"][0]["name"]
        id = id[1:]
        return id, name

    def get_blog_url(self, id):
        url = f"https://api.place.naver.com/place/graphql"
        query = [
            {
                "operationName": "getFsasReviews",
                "variables": {
                    "input": {
                        "businessId": str(id),
                        "businessType": "place",
                        "page": 0,
                        "display": 10,
                        "deviceType": "mobile",
                        "query": None,
                    }
                },
                "query": "query getFsasReviews($input: FsasReviewsInput) {\n  fsasReviews(input: $input) {\n    ...FsasReviews\n    __typename\n  }\n}\n\nfragment FsasReviews on FsasReviewsResult {\n  total\n  maxItemCount\n  items {\n    name\n    type\n    typeName\n    url\n    home\n    id\n    title\n    rank\n    contents\n    bySmartEditor3\n    hasNaverReservation\n    thumbnailUrl\n    thumbnailUrlList\n    thumbnailCount\n    date\n    isOfficial\n    isRepresentative\n    profileImageUrl\n    isVideoThumbnail\n    reviewId\n    authorName\n    createdString\n    __typename\n  }\n  __typename\n}\n",
            }
        ]
        response = json.loads(RequestBlog().post(url, query).text)
        total = response[0]["data"]["fsasReviews"]["total"]
        urls = self.get_url_list(response)

        i = 0
        start = time.time()

        while True:
            i += 1
            now = time.time()
            print(f"\r{i+1} runtime: {now - start:.2f}", end="")
            query[0]["variables"]["input"]["page"] = i
            response = json.loads(RequestBlog().post(url, query).text)
            url_list = self.get_url_list(response)
            if not url_list:
                break
            urls.extend(self.get_url_list(response))
        print("\n")
        return urls

    def get_url_list(self, response):
        urls = []
        if len(response[0]["data"]["fsasReviews"]["items"]) == 0:
            return None
        else:
            for blog in response[0]["data"]["fsasReviews"]["items"]:
                urls.append(blog["url"])
        return urls

    def get_context(self, url):
        response = RequestBlog().get(url)
        soup = BeautifulSoup(response.text, "lxml")
        if soup.find("div", attrs={"class": "se-component se-text se-l-default"}):
            contexts = soup.find_all(
                "div", attrs={"class": "se-component se-text se-l-default"}
            )

            def map_fun(x):
                return x.get_text().replace("\n", "").replace("\u200b", "")

            contexts = list(map(map_fun, contexts))
            text = " ".join(contexts)
            text = ReplaceText().get_convert_text(text)
            return text, url
        elif soup.find("div", attrs={"class": "post_ct"}):
            context = soup.find("div", attrs={"class": "post_ct"}).get_text()
            text = ReplaceText().get_convert_text(context)
            return text, url
        elif re.findall(
            r"""(<(p|h[0-7]) class="se_textarea".*<\/(p|h[0-7])>)""", response.text
        ):
            contexts = re.findall(
                r"""(<(p|h[0-7]) class="se_textarea".*<\/(p|h[0-7])>)""", response.text
            )

            def map_fun(x):
                return re.sub(r"""<[^>]*>""", "", x[0])

            contexts = list(map(map_fun, contexts))
            context = " ".join(contexts)

            text = ReplaceText().get_convert_text(context)
            return text, url
        # 카페 처리 부분
        elif re.findall(r"""http:\/\/cafe.naver.com.*""", url):
            return None, url
        else:
            print(f"\nError {url}")
            return None, url


if __name__ == "__main__":
    start = time.time()
    type_list = ["관광지", "문화시설", "행사/공연/축제", "레포츠"]
    file_path = os.getenv("DATA_GEN_DATA_PATH")
    with open(file_path + "tour_spot_name.json", "r") as f:
        tour_name = json.load(f)

    info = {}
    for state in tour_name.keys():
        info[state] = {}
        for types in type_list:
            info[state][types] = {}
            for i, location in enumerate(tour_name[state][types].keys()):
                if i == 0:
                    continue
                crawling = Crawling_naver(location)
                result, info_ = crawling.get_result()
                info[state][types][location] = info_

    now = time.time()
    print(f"runtime: {now - start:.2f}")

    with open(file_path + "test.json", "w", encoding="utf-8-sig") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
