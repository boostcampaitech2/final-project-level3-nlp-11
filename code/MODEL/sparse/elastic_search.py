import json
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import time
from subprocess import Popen, PIPE, STDOUT
import re
import pandas as pd
from datasets import Dataset, DatasetDict, Value, Features
import sys, os

class ElasticSearch:
    def __init__(self, dir_path:str = "/opt/ml/final-project-level3-nlp-11", index_name:str="wiki-1"):
        self.config: dict = {"host": "localhost", "port": 9200}
        self.dir_path = dir_path

        self.index_name = index_name
        self.get_place_data()
        self.es = self.set_elastic_server()
        if not self.es:
            exit()
        is_already_exist = self.set_mapping()
        if not is_already_exist:
            self.insert_data_to_elastic()
    
    def get_place_data(self):
        with open(f'{self.dir_path}/data/MODEL/pair.json', "r", encoding='utf-8-sig') as f:
            blog_info = json.load(f)
            self.contexts = []
            for area in blog_info:
                for location in blog_info[area]['관광지']:
                    for pair in blog_info[area]['관광지'][location]:
                        context = pair["context"]
                        self.contexts.append({'context':context, 'place':location})

    def set_elastic_server(self):
        path_to_elastic = f"{self.dir_path}/code/MODEL/sparse/elasticsearch-7.9.2/bin/elasticsearch"
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

    def set_mapping(self) -> bool:
        if self.es.indices.exists(index=self.index_name):
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
                    },
                },
            }

            print(
                self.es.indices.create(
                    index=self.index_name, body=index_config, ignore=400
                )
            )
            return False

    def insert_data_to_elastic(self) -> None:
        for i, rec in enumerate(tqdm(self.contexts)):
            try:
                index_status = self.es.index(index=self.index_name, id=i, body=rec)
            except Exception as e:
                print(f"Unable to load document {i}. because of {e}")
        n_records = self.es.count(index=self.index_name)["count"]
        print(f"Succesfully loaded {n_records} into {self.index_name}")

    def get_top_k_passages(self, describe: str, k: int) -> list:
        query = {"query": {"match": {"context": describe}}}
        result = self.es.search(index=self.index_name, body=query, size=k)
        return result["hits"]["hits"]

    def run_retrieval(self, describe: str, topk: int=50) -> DatasetDict:
        results = []
        for idx, q_data in enumerate(q_dataset):
            describe = q_data["describe"]
            passages = self.get_top_k_passages(describe=describe, k=topk)
            examples = [{"context": passages[i]["_source"]["context"], "place":passages[i]["_source"]["place"], "query": describe, "score":passages[i]["_score"]} for i in range(len(passages))]
            results.extend(examples)

        passages = self.get_top_k_passages(describe=describe, k=topk)
        results = [{"id":int(passages[i]["_id"]), "context": passages[i]["_source"]["context"], "place":passages[i]["_source"]["place"], "score":passages[i]["_score"]} for i in range(len(passages))]

        df = pd.DataFrame(results)

        return df
