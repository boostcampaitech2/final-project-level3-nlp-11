import sys
import os
import pickle
import json

from typing import List, Tuple, NoReturn, Any, Optional, Union
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    AutoModel,
    AutoTokenizer,
)

# /opt/ml/final-project-level3-nlp-11/code/MODEL/sparse/elastic_search.py
sys.path.append("/opt/ml/final-project-level3-nlp-11/code/MODEL/sparse")
from elastic_search import ElasticSearch

class DenseRetrieval:
    def __init__(
        self,
        p_encoder_model:str,
        q_encoder_model:str,
        tokenizer_name:str,
        pickle_path:str,
        token_length:int,
        es:ElasticSearch=None,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.p_encoder = AutoModel.from_pretrained(p_encoder_model).to(self.device)
        self.q_encoder = AutoModel.from_pretrained(q_encoder_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.es = es
        self.pickle_path = pickle_path
        self.token_length = token_length
        
        self.p_embedding = None
        self.set_contexts()
        
        torch.cuda.empty_cache()
        
    def set_contexts(self):
        self.contexts = []
        self.places = []
        self.area_idx = {}
        with open("../../data/pair.json", "r", encoding="utf-8-sig") as f:
            location_list = json.load(f)
            change_area = {"대전" : "충청남도", "세종특별자치시": "충청남도", "울산" : "경상남도", "광주" : "전라남도"}
            new_area_list = {}
            for area in location_list:
                if area in change_area:
                    aft_area_name = change_area[area]
                    if aft_area_name in new_area_list:
                        new_area_list[aft_area_name].append(area)
                    else:
                        new_area_list[aft_area_name] = [area]
                else:
                    if area in new_area_list:
                        new_area_list[area].append(area)
                    else:
                        new_area_list[area] = [area]
            start_idx = 0
            end_idx = 0
            for area_list in new_area_list:
                for area in new_area_list[area_list]:
                    for location in location_list[area]['관광지']:
                        for pair in location_list[area]['관광지'][location]:
                            self.contexts.append(pair['context'])
                            self.places.append(location)
                            end_idx += 1
                self.area_idx[area_list] = [start_idx, end_idx]
                start_idx = end_idx
            self.area_idx["전국"] = [0, end_idx]
                
    
    def get_embedding(self):
        emd_path = self.pickle_path

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)

            print("Embedding pickle load.")
        else:
            print("Embedding pickle is not existed")
            print("save_embedding() exec")
            self.save_embedding(emd_path)
    
    def save_embedding(self, save_path: str):
        contexts = self.contexts

        tokenizer = self.tokenizer
        p_encoder = self.p_encoder
        p_embs = []

        with torch.no_grad():
            p_encoder.eval()
            for p in tqdm(contexts):
                p = tokenizer(
                    p, padding="max_length", max_length=self.token_length, truncation=True, return_tensors="pt"
                ).to(self.device)
                p_emb = p_encoder(**p).pooler_output.to("cpu").detach().numpy()
                p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()

        p_embedding = p_embs
        self.p_embedding = p_embedding
        with open(save_path, "wb") as file:
            pickle.dump(p_embedding, file)
        print("Embedding pickle saved.")
        
    def inference(self, single_query, topk:int=25, area:str="전국", use_elastic=True):
        """
            area : 명소 검색 시 선택된 드랍다운 지역
            use_elastic : elastic + dense로 리트리빙을 진행할 지, dense만을 이용하여 진행할지 선택
        """
        if self.p_embedding == None:
            self.get_embedding()
        
        if use_elastic:
            doc_scores, doc_indices = self.get_relevant_doc_elastic(
                single_query, k=topk, area=area
            )
        else:
            doc_scores, doc_indices = self.get_relevant_doc(
                single_query, k=topk, area=area
            )
        total = []

        for i in range(topk):
            # print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
            # print('Content name :', self.places[doc_indices[i]])
            # print('Contexts :', self.contexts[doc_indices[i]])
            tmp = {
                # Retrieve한 Passage의 id, context를 반환합니다.
                "rank": i,
                "scores": doc_scores[i],
                "place": self.places[doc_indices[i]],
                "context_id": doc_indices[i],
                "context": self.contexts[doc_indices[i]]
            }
            total.append(tmp)
        pred_places = pd.DataFrame(total)
        return pred_places
    
    def get_relevant_doc(self, query: str, area:str, k:int = 5):
        """
            area : 명소 검색 시 선택된 드랍다운 지역

            지역에 따른 index 정보를 이용하여 self.p_embedding에서 
            해당 area의 embedding만을 p_embs에 담음.

            sorted_result는 p_embs 기준의 index값을 감고 있기 때문에, 
            self.p_embedding기준의 index값을 doc_indices가 리턴할 수 있도록
            start_idx를 sorted_result의 값들에 더하여 이용
        """
        q_encoder = self.q_encoder
        if area == "전국":
            p_embs = self.p_embedding
            start_idx = 0
        else:
            start_idx, end_idx = self.area_idx[area]
            p_embs = self.p_embedding[start_idx:end_idx]

        with torch.no_grad():
            q_encoder.eval()
            print('getting Query X Passage scores')
            q = self.tokenizer(
                query, max_length=self.token_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = q_encoder(**q).pooler_output.to("cpu").detach().numpy()

        # result = q_emb * self.p_embedding.T
        q_emb = np.array(q_emb)
        q_emb = torch.Tensor(q_emb).squeeze()
        result = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

        if not isinstance(result, np.ndarray):
            result = np.array(result.tolist())

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = [i+start_idx for i in sorted_result.tolist()[:k]]
        return doc_score, doc_indices

    def get_relevant_doc_elastic(self, query: str, area: str, k: int = 5):
        q_encoder = self.q_encoder
        p_embs, p_embs_index_list = self.es_run_retrieval(query, area)
        if k > len(p_embs):
            return self.get_relevant_doc(
                query, k=k, area=area
            )

        with torch.no_grad():
            q_encoder.eval()
            print('getting Query X Passage scores')
            q = self.tokenizer(
                query, max_length=self.token_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            q_emb = q_encoder(**q).pooler_output.to("cpu").detach().numpy()

        # result = q_emb * self.p_embedding.T
        q_emb = np.array(q_emb)
        q_emb = torch.Tensor(q_emb).squeeze()
        result = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

        if not isinstance(result, np.ndarray):
            result = np.array(result.tolist())

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        
        embs_indices = sorted_result.tolist()[:k]
        doc_indices = [p_embs_index_list[i] for i in embs_indices]
        return doc_score, doc_indices
    
    def es_run_retrieval(self, query: str, area: str):
        """
            elastic search로 먼저 top 100~200 추출하는 함수
        """
        es_result = self.es.run_retrieval(describe=query, index_name=area)
        result_emb_list = []
        start_idx = self.area_idx[area][0]
        for data in es_result.iterrows():
            context_id = start_idx + data[1]["id"]
            result_emb_list.append(context_id)
        return self.p_embedding[result_emb_list], result_emb_list
    
        

if __name__ == '__main__':
    """ test code 
        실행하기 전, elasticsearch.sh를 먼저 실행해주세요
        pair.json의 위치를 /data/ 로 해주세요 
        pickle_path를 파일명으로 지정해주세요
        """
    es = ElasticSearch()
    base_model = "klue/bert-base"
    retrieval = DenseRetrieval(p_encoder_model=base_model, q_encoder_model=base_model, tokenizer_name=base_model, pickle_path="../MODEL/embedding/embedding.pkl", token_length=128, es=es)
    query = "기분이 울적할 때 갈만한 곳"
    result = retrieval.inference(query)
    result.to_csv("result.csv", index=False)