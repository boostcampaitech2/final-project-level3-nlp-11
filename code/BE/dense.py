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
        es:ElasticSearch,
    ) -> None:
        self.p_encoder = AutoModel.from_pretrained(p_encoder_model)
        self.q_encoder = AutoModel.from_pretrained(q_encoder_model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.es = es
        self.pickle_path = pickle_path
        self.token_length = token_length
        
        self.p_embedding = None
        self.contexts = []
        self.places = []
        with open("../../data/pair.json", "r", encoding="utf-8-sig") as f:
            location_list = json.load(f)
            for area in location_list:
                for location in location_list[area]['관광지']:
                    for pair in location_list[area]['관광지'][location]:
                        self.contexts.append(pair['context'])
                        self.places.append(location)
        torch.cuda.empty_cache()
    
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
        
    def inference(self, single_query, topk=5):
        if self.p_embedding == None:
            self.get_embedding()
        doc_scores, doc_indices = self.get_relevant_doc(
            single_query, k=topk
        )
        total = []
        for i in range(topk):
            print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
            print('Content name :', self.places[doc_indices[i]])
            print('Contexts :', self.contexts[doc_indices[i]])
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

    def get_relevant_doc(self, query: str, k = 5):
        q_encoder = self.q_encoder
        q_embs = []
        p_embs, p_embs_index_list = self.es_run_retrieval(query)

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
    
    def es_run_retrieval(self, query: str):
        es_result = es.run_retrieval(query)
        result_emb_list = []
        for data in es_result.iterrows():
            context_id = data[1]["id"]
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