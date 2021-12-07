import os
import urllib.parse
import json
import time
import argparse
import warnings
from src.crawling.utils.request_blog import RequestBlog
from src.crawling.utils.blog_parser import BlogParser
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class Crawling_naver:
    def __init__(self, args) -> None:

        start = time.time()
        type_list = ["관광지", "문화시설", "행사/공연/축제", "레포츠"]
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        start_point = {
            "state": args.start_state,
            "type": args.start_type,
            "location": args.start_location,
            "flag": args.use_start_point,
        }
        self.mk_file(type_list, file_path, start_point, args)
        now = time.time()
        print(f"runtime: {now - start:.2f}")

    def mk_file(self, type_list, file_path, start_point, args):
        with open(file_path + args.name_path, "r", encoding="utf-8-sig") as f:
            try:
                tour_name = json.load(f)
            except:
                print("error need tour_spot_name data")
                exit(1)
        if not os.path.isfile(file_path + args.info_path):
            f = open(file_path + args.info_path, "w")
            f.close()
        with open(file_path + args.info_path, "r", encoding="utf-8-sig") as f:
            try:
                info = json.load(f)
            except:
                info = {}

        for state in tour_name.keys():
            if state != start_point["state"] and start_point["flag"]:
                continue
            if not state in info:
                info[state] = {}
            for types in type_list:
                if types != start_point["type"] and start_point["flag"]:
                    continue
                if not types in info[state]:
                    info[state][types] = {}
                for i, location in enumerate(tour_name[state][types].keys()):
                    if i == 0:
                        continue
                    elif location != start_point["location"] and start_point["flag"]:
                        continue
                    elif location == start_point["location"] and start_point["flag"]:
                        start_point["flag"] = False
                        continue
                    self.query = location
                    result, info_ = self.get_result()
                    info[state][types][location] = info_
                    if result:
                        tour_name[state][types][location].extend(result)

                    with open(
                        file_path + args.info_path, "w", encoding="utf-8-sig"
                    ) as f:
                        json.dump(info, f, indent=4, ensure_ascii=False)

                    with open(
                        file_path + args.result_path, "w", encoding="utf-8-sig"
                    ) as f:
                        json.dump(tour_name, f, indent=4, ensure_ascii=False)

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
            contexts.append({"context": text, "url": url, "type": "blog"})

        # contexts = list(filter(None,contexts))
        info["total"] = len(contexts)
        print("\n")
        return contexts, info

    def get_id(self, query):

        query_ = urllib.parse.quote_plus(query)

        url = f"https://m.map.naver.com/search2/searchMore.naver?query={query_}&sm=clk&style=v5&page=1&displayCount=75&type=SITE_1"

        response = json.loads(RequestBlog().get(url).text)

        if not response["result"]:
            print(f'query:{query} "정보 없음"')
            id = -1
            return id, None
        elif not "site" in response["result"]:
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
        if not urls:
            return urls
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
            urls.extend(url_list)
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
        return BlogParser(response, url).get_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_state", type=str, default="서울", help="set start_point")
    parser.add_argument("--start_type", type=str, default="관광지", help="set start_point")
    parser.add_argument(
        "--start_location", type=str, default="간데메공원", help="set start_point"
    )
    parser.add_argument(
        "--use_start_point", type=bool, default=False, help="use start_point"
    )
    parser.add_argument(
        "--name_path", type=str, default="tour_spot_name.json", help="set start_point"
    )
    parser.add_argument(
        "--info_path", type=str, default="test_info.json", help="set start_point"
    )
    parser.add_argument(
        "--result_path", type=str, default="test_result.json", help="set start_point"
    )
    args = parser.parse_args()
    Crawling_naver(args)
