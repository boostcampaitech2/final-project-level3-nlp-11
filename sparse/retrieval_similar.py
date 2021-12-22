import os
import json
import time
import argparse
import pickle
import numpy as np
import pandas as pd

from typing import Optional, NoReturn, List, Tuple, Set

from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


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

    def json_to_list(self):
        """
        Return : list of contexts and contents
        Only for content type of '관광지'
        """
        with open(
            os.path.join(self.data_path, self.json_name), "r", encoding="utf-8-sig"
        ) as f:
            tour_json = json.load(f)

        tour_context = []
        tour_content = []
        tour_area_idx = {}  ##area:[start_idx, end_idx]

        area_list = ['서울', '인천', '대구', '부산', '경기도', '강원도', '충청북도', '충청남도', '대전', 
             '세종특별자치시', '경상북도', '경상남도', '울산', '전라북도', '전라남도', '광주', '제주도']
        
        cnt_idx = 0
        for area in area_list:
            loc_list = list(tour_json[area]["관광지"].keys())
            tour_area_idx[area] = [cnt_idx]  ##start idx

            for loc in loc_list:
                pairs = tour_json[area]["관광지"][loc]

                for pair in pairs:
                    tour_context.append(pair["context"])
                    tour_content.append(loc)
                    cnt_idx += 1

            tour_area_idx[area].append(cnt_idx)  ##end idx
            
        ## Merge area under 30 contents
        tour_area_idx['충청남도'][1] = tour_area_idx['세종특별자치시'][1]
        tour_area_idx['경상남도'][1] = tour_area_idx['울산'][1]
        tour_area_idx['전라남도'][1] = tour_area_idx['광주'][1]

        del tour_area_idx['대전']
        del tour_area_idx['세종특별자치시']
        del tour_area_idx['울산']
        del tour_area_idx['광주']
        ## Merge area under 30 contents

        self.contexts = tour_context
        self.contents = tour_content
        self.area_idx = tour_area_idx

    def get_precision(self, query, content) -> float:
        """
        Input
        Output
        """
        query = query.replace(" ", "")
        content = content.replace(" ", "")

        cnt = 0
        for char in query:
            if char in content:
                cnt += 1
        return cnt / len(query)

    def get_query_contexts(self, query) -> bool:
        """
        return True : found matching content.
        return False : there's no matching content, use query as a sentence to dense.
        """
        ### Get matched content over 0.8 precision score
        top_score = 0
        chosen = False
        for content in set(self.contents):
            prec_score = self.get_precision(query, content)
            if prec_score > 0.8 and prec_score > top_score:
                top_score = prec_score
                chosen = content
        ### Get matched content over 0.8 precision score

        ### Nothing matches
        if not chosen:
            return chosen
        ### Nothing matches

        ### Get context of content
        chosen_idx = self.contents.index(chosen)
        chosen_context = self.contexts[chosen_idx : chosen_idx + 5]
        ### Get context of content

        self.query_idx = chosen_idx
        self.query_contexts = chosen_context

        return True

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

    def retrieve(
        self, query_contexts: List, topk: int, area: Optional[str] = None
    ) -> Set[str]:

        doc_scores, doc_indices, contents_exc = self.get_relevant_doc_bulk(
            query_contexts, k=topk, area=area
        )
        result_contents = set()
        break_idx = 0

        for rank_idx in range(topk):
            for idx in range(len(query_contexts)):
                result_contents.add(contents_exc[doc_indices[idx][rank_idx]])

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
        self, queries: List, k: Optional[int] = 1, area: Optional[str] = None
    ) -> Tuple[List, List, List]:

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
        ## search for whole area
        if area == '전국':
            delete_idx = list(range(self.query_idx, self.query_idx + 5))
            result = np.delete(result, delete_idx, 1)
            contents_exc = np.array(self.contents)
            contents_exc = np.delete(contents_exc, delete_idx, 0)
        ## search for whole area
        
        ## getting area indices, exclude query indices
        else:
            start_idx = self.area_idx[area][0]
            end_idx = self.area_idx[area][1]
            area_indices = list(range(start_idx, end_idx))

            result = np.take(result, area_indices, 1)
            contents_exc = np.array(self.contents)
            contents_exc = np.take(contents_exc, area_indices, 0)

            if self.query_idx in area_indices:
                start_query_idx = self.query_idx - start_idx
                delete_idx = list(range(start_query_idx, start_query_idx + 5))

                result = np.delete(result, delete_idx, 1)
                contents_exc = np.delete(contents_exc, delete_idx, 0)
        ## getting area indices, exclude query indices

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            ## sort for top k
            sorted_result = np.argsort(result[i])[::-1]  # Sorting Indices

            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
            ## sort for top k

        return doc_scores, doc_indices, contents_exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_path", default="../data", type=str, help="dataset directory path"
    )
    parser.add_argument(
        "--json_name", default="pair.json", type=str, help="json dataset path"
    )
    parser.add_argument("--topk", default=5, type=int, help="number of k for top-k")
    parser.add_argument(
        "--pre_tokenizer",
        default="klue/bert-base",
        type=str,
        help="transformer's model name for tokenizer",
    )
    args = parser.parse_args()

    t0 = time.time()

    query = "해운대온천"
    area = '전라남도'
    retriever = SimilarSparse(
        args.pre_tokenizer,
        data_path=args.data_path,
        json_name=args.json_name,
    )
    retriever.get_sparse_embedding()

    t0 = time.time()
    true_false = retriever.get_query_contexts(query)
    if not true_false:
        print("No Matches, Run dense with query")
        exit()
    else:
        result_contents = retriever.retrieve(retriever.query_contexts, 5, area)

    #     ## TEST
    #     tour_context = retriever.contexts
    #     tour_content = retriever.contents

    #     chosen_idx = retriever.query_idx
    #     chosen_contexts = retriever.query_contexts

    #     print(len(tour_context), len(tour_content))
    #     print(len(set(tour_content)))
    #     print(tour_content[chosen_idx:chosen_idx + 5])
    #     ## TEST

    ## RESULT
    print(retriever.contents[retriever.query_idx])
    print(result_contents)
    print(time.time() - t0)
    ## RESULT
