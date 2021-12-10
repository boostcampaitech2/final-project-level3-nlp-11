import os
import json
import time
import argparse
import pickle
import numpy as np
import pandas as pd

from typing import Optional, NoReturn, List, Tuple

from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from contextlib import contextmanager

# @contextmanager
# def timer(name):
#     t0 = time.time()
#     yield
#     print(f"[{name}] done in {time.time() - t0:.3f} s")

class SimilarSparse:
    def __init__(
        self,
        pre_tokenizer,
        data_path: str,
        json_name: str,
    ) -> NoReturn:
        tokenizer = AutoTokenizer.from_pretrained(
            pre_tokenizer,
            use_fast=False,
        )
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenizer.tokenize,
            ngram_range=(1, 2), 
        )
        
        self.data_path = data_path
        self.json_name = json_name
        
        self.json_to_list()
        
        
    def json_to_list(self, area: Optional[str] = None):
        '''
        Return : list of contexts and contents
        Only for content type of '관광지'
        '''
        with open(os.path.join(self.data_path, self.json_name), "r", encoding="utf-8-sig") as f:
            tour_json = json.load(f)

        tour_context = []
        tour_content = []

        if area:
            loc_list = list(tour_json[area]['관광지'].keys())

            for loc in loc_list:
                pairs = tour_json[area]['관광지'][loc]
                for pair in pairs:
                    tour_context.append(pair['context'])
                    tour_content.append(loc)

        else:
            area_list = list(tour_json.keys())
            for area in area_list:
                loc_list = list(tour_json[area]['관광지'].keys())
                for loc in loc_list:
                    pairs = tour_json[area]['관광지'][loc]
                    for pair in pairs:
                        tour_context.append(pair['context'])
                        tour_content.append(loc)

        self.contexts = tour_context
        self.contents = tour_content

    def get_precision(self, query, content):
        '''
        Input
        Output
        '''
        query = query.replace(' ','')
        content = content.replace(' ','')

        cnt = 0
        for char in query:
            if char in content:
                cnt += 1
        return cnt / len(query)
    
    def get_query_contexts(self, query):
        ### Get matched content over 0.8 precision score
        top_score = 0
        for content in set(self.contents):
            prec_score = self.get_precision(query, content)
            if prec_score > 0.8 and prec_score > top_score:
                top_score = prec_score
                chosen = content
        ### Get matched content over 0.8 precision score

        ### Get context of content
        chosen_idx = self.contents.index(chosen)
        chosen_context = self.contexts[chosen_idx:chosen_idx + 5]
        ### Get context of content
        
        self.query_idx = chosen_idx
        self.query_contexts = chosen_context
    
    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding_tour.bin"
        tfidfv_name = f"tfidv_tour.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")
            
    def retrieve(self, query_contexts: List, topk: int) -> pd.DataFrame:

        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            query_contexts, k=topk
        )
        result_contents = set()
        break_idx = 0
        
        for rank_idx in range(topk):
            for idx in range(len(query_contexts)):
                result_contents.add(self.contents[doc_indices[idx][rank_idx]])
                if len(result_contents) == topk:
                    break_idx = 1
                    break
            if break_idx == 1:
                break
                
        return result_contents
        
        # ## For TEST
        # total = []
        # for idx in range(len(query_contexts)):
        #     tmp = {
        #         "context_id": doc_indices[idx],
        #         "context": " ".join(
        #             [self.contexts[pid] for pid in doc_indices[idx]]
        #         ),
        #     }
        #     total.append(tmp)
        # cqas = pd.DataFrame(total)
        # ## For TEST
        # return cqas # for TEST

    
    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
            
        ## exclude query indices
        delete_idx = list(range(self.query_idx, self.query_idx + 5))
        result = np.delete(result, delete_idx, 1)
        ## exclude query idices
        
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            ## Search for all area
            sorted_result = np.argsort(result[i])[::-1]#Sorting Indices

            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
            ## Search for all area
            
            ## Search for specific area
            
            ## Search for specific area
            
        return doc_scores, doc_indices
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_path", default="../data", type=str, help="dataset directory path"
    )
    parser.add_argument(
        "--json_name", default="pair.json", type=str, help="json dataset path"
    )
    parser.add_argument(
        "--topk", default=5, type=int, help="number of k for top-k"
    )
    parser.add_argument(
        "--pre_tokenizer",
        default="klue/bert-base",
        type=str,
        help="transformer's model name for tokenizer",
    )
    args = parser.parse_args()
    
    t0 = time.time()
    
    query = '효자동골목'
    retriever = SimilarSparse(
        args.pre_tokenizer,
        data_path=args.data_path,
        json_name=args.json_name,
    )

    retriever.get_query_contexts(query)
    retriever.get_sparse_embedding()
    result_df = retriever.retrieve(retriever.query_contexts, 5)
    
#     ## TEST
#     tour_context = retriever.contexts
#     tour_content = retriever.contents
    
#     chosen_idx = retriever.query_idx
#     chosen_contexts = retriever.query_contexts
    
#     print(len(tour_context), len(tour_content))
#     print(len(set(tour_content)))
#     print(tour_content[chosen_idx:chosen_idx + 5])
    
#     print(retriever.contents[retriever.query_idx])
#     print(result_df)
#     ## TEST
    
    ## RESULT
    print(retriever.contents[retriever.query_idx])
    print(result_df)
    print(time.time() - t0)
    ## RESULT