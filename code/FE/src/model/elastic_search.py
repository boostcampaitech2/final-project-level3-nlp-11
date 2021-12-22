import json
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import time
from subprocess import Popen, PIPE, STDOUT
import re
import pandas as pd
from datasets import Dataset, DatasetDict, Value, Features
import sys, os

from dotenv import load_dotenv


env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)

home_path = os.getenv("HOME_PATH")


class ElasticSearch:
    def __init__(self, dir_path: str = "/opt/ml/final-project-level3-nlp-11"):
        self.config: dict = {"host": "localhost", "port": 9200}
        self.dir_path = dir_path

        self.get_place_data()
        self.es = self.set_elastic_server()
        if not self.es:
            exit()

    def get_place_data(self):
        with open(
            f"{self.dir_path}/data/Sparse/pair.json", "r", encoding="utf-8-sig"
        ) as f:
            place_info = json.load(f)
            self.contexts = {"전국": []}
            change_area = {"대전": "충청남도", "세종특별자치시": "충청남도", "울산": "경상남도", "광주": "전라남도"}
            i = 0

            for area in place_info:
                if area in change_area:
                    aft_area = change_area[area]
                    if aft_area not in self.contexts:
                        self.contexts[aft_area] = []
                    j = len(self.contexts[aft_area])
                else:
                    if area not in self.contexts:
                        self.contexts[area] = []
                    j = len(self.contexts[area])

                for location in place_info[area]["관광지"]:
                    for pair in place_info[area]["관광지"][location]:
                        context = pair["context"]
                        self.contexts["전국"].append(
                            {"context": context, "place": location, "doc_id": i}
                        )
                        i += 1
                        if area in change_area:
                            self.contexts[aft_area].append(
                                {"context": context, "place": location, "doc_id": j}
                            )
                        else:
                            self.contexts[area].append(
                                {"context": context, "place": location, "doc_id": j}
                            )
                        j += 1

    def set_elastic_server(self):
        path_to_elastic = (
            f"{self.dir_path}/code/MODEL/sparse/elasticsearch-7.9.2/bin/elasticsearch"
        )
        es_server = Popen(
            [path_to_elastic],
            stdout=PIPE,
            stderr=STDOUT,
            preexec_fn=lambda: os.setuid(1),  # as daemon
        )
        config = {"host": "localhost", "port": 9200}
        print("You have to wait 20secs for connecting to elastic server")
        for _ in tqdm(range(20)):
            time.sleep(1)
        es = Elasticsearch([config], timeout=30)
        ping_result = es.ping()
        if ping_result:
            print("Connecting Success !!")
            return es
        else:
            print(
                "Connecting Failed.. You have to try again. You have to follow class description"
            )
            return None

    def set_mapping(self, index_name: str = "전국") -> bool:
        if self.es.indices.exists(index=index_name):
            print("Index Mapping already exists.")
            return True
        else:
            index_config = {
                "settings": {
                    "analysis": {
                        "filter": {
                            "my_stop_filter": {
                                "type": "stop",
                                "stopwords_path": "my_stop_dic.txt",
                            }
                        },
                        "analyzer": {
                            "nori_analyzer": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "decompound_mode": "mixed",
                                "filter": ["my_stop_filter"],
                            }
                        },
                    }
                },
                "mappings": {
                    "dynamic": "strict",
                    "properties": {
                        "context": {"type": "text", "analyzer": "nori_analyzer"},
                        "place": {"type": "text", "analyzer": "nori_analyzer"},
                        "doc_id": {"type": "text"},
                    },
                },
            }

            print(
                self.es.indices.create(index=index_name, body=index_config, ignore=400)
            )
            return False

    def insert_data_to_elastic(self, index_name: str) -> None:
        for i, rec in enumerate(tqdm(self.contexts[index_name])):
            try:
                index_status = self.es.index(index=index_name, id=i, body=rec)
            except Exception as e:
                print(f"Unable to load document {i}. because of {e}")
        n_records = self.es.count(index=index_name)["count"]
        print(f"Succesfully loaded {n_records} into {index_name}")

    def get_top_k_passages(self, describe: str, k: int, index_name: str = "전국") -> list:
        query = {"query": {"match": {"context": describe}}}
        result = self.es.search(index=index_name, body=query, size=k)
        return result["hits"]["hits"]

    def run_retrieval(
        self, describe: str, topk: int = 100, index_name: str = "전국"
    ) -> DatasetDict:
        results = []

        is_already_exist = self.set_mapping(index_name=index_name)
        if not is_already_exist:
            self.insert_data_to_elastic(index_name=index_name)
        passages = self.get_top_k_passages(
            describe=describe, k=topk, index_name=index_name
        )
        results = [
            {
                "id": int(passages[i]["_source"]["doc_id"]),
                "context": passages[i]["_source"]["context"],
                "place": passages[i]["_source"]["place"],
                "score": passages[i]["_score"],
            }
            for i in range(len(passages))
        ]

        df = pd.DataFrame(results)

        return df
