import os
import pickle
import argparse
import json
from tqdm.auto import tqdm

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from dense_model import BertEncoder
# from datasets import load_from_disk


class RetrievalInference:
    def __init__(
        self, args, q_encoder: BertEncoder, tokenizer: AutoTokenizer, context_path: str
    ):
        self.pickle_path = args.pickle_path
        self.q_encoder = q_encoder
        self.tokenizer = tokenizer

        # with open(context_path, "r", encoding="utf-8") as f:
        #     wiki = json.load(f)
        # self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

## New json to context
        with open(args.dataset_name, "r", encoding="utf-8-sig") as f:
            train_data = json.load(f)

        area_list = list(train_data.keys())
        loc_type_list = list(train_data[area_list[0]].keys()) # Same type shared
        self.context = []
        self.query = []
        self.content = []
        
        for area in area_list:
            loc_list = list(train_data[area]['관광지'].keys())
            
            for loc in loc_list:
                pairs = train_data[area]['관광지'][loc]

                for pair in pairs:
                    self.query.append(pair['query'])
                    self.context.append(pair['context'])
                    self.content.append(loc)
## New json to context
        
    def get_dense_embedding(self):
        emb_path = self.pickle_path

        if os.path.isfile(emb_path):
            with open(emb_path, "rb") as file:
                self.p_embedding = pickle.load(file)

            print("Embedding pickle load.")
            
    def retrieval(self, single_query, topk=1):
        assert self.p_embedding is not None
        
        if single_query:
            doc_scores, doc_indices = self.get_relevant_doc(
                single_query, k=topk
            )
            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print('Content name :', self.content[doc_indices[i]])
                print(self.context[doc_indices[i]])        
        else:
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                self.query, k=topk
            )
            for idx, example in enumerate(tqdm(self.query, desc="Dense retrieval: ")):
                context_array = []
                for pid in doc_indices[idx]:
                    context = "".join(self.context[pid])
                    context_array.append(context)
                tmp = {
                    "question": example,
                    "context_id": doc_indices[idx],
                    "context": context_array,
                    "scores": doc_scores[idx],
                }
                tmp['answers'] = self.context[idx]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k = 1):
        q_encoder = self.q_encoder
        q_embs = []

        with torch.no_grad():
            q_encoder.eval()
            print('getting Query X Passage scores')
            q = tokenizer(
                query, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = q_encoder(**q).to("cpu").detach().numpy()

        # result = q_emb * self.p_embedding.T
        q_emb = np.array(q_emb)
        q_emb = torch.Tensor(q_emb).squeeze()
        result = torch.matmul(q_emb, torch.transpose(self.p_embedding, 0, 1))

        if not isinstance(result, np.ndarray):
            result = np.array(result.tolist())

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices
        
    def get_relevant_doc_bulk(self, queries: list, k=1):
        q_encoder = self.q_encoder
        q_embs = []

        with torch.no_grad():
            q_encoder.eval()
            print('getting Query X Passage scores')
            for q in tqdm(queries):
                q = tokenizer(
                    q, padding="max_length", truncation=True, return_tensors="pt"
                ).to("cuda")
                q_emb = q_encoder(**q).to("cpu").detach().numpy()
                q_embs.append(q_emb)

        q_embs = np.array(q_embs)
        q_embs = torch.Tensor(q_embs).squeeze()
        result = torch.matmul(q_embs, torch.transpose(self.p_embedding, 0, 1))

        if not isinstance(result, np.ndarray):
            result = np.array(result.tolist())
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices

    def get_acc_score(self, df: pd.DataFrame):
        df["correct"] = False
        df["correct_rank"] = 0
        for i in tqdm(range(len(df)), desc="check tok_n"):
            count = 0
            for context in df.iloc[i]["context"]:
                count += 1
                if df.iloc[i]["answers"] == context:
                    df.at[i, "correct"] = True
                    df.at[i, "correct_rank"] = count

        return df

    def print_result(self, df: pd.DataFrame, length: int):
        # for i in range(length):
        #     print("=======================================")
        #     f = df.iloc[i]
        #     print(f'Question         : {f["question"]}')
        #     print(f'answers : {f["answers"]}')
        #     print("\n\n")
        #     for i in range(len(f["context"])):
        #         print(f'score\t:{f["scores"][i]},\ncontext\t: {f["context"][i]}\n')
        #     print("=======================================")

        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/pair.json", type=str, help=""
    )
    parser.add_argument(
        "--tokenizer_name",
        default="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument(
        "--pickle_path",
        default="../data/dense/dense_embedding.bin.bin",
        type=str,
        help="wiki embedding path",
    )
    parser.add_argument(
        "--context_path",
        default="../data/pair.json",
        type=str,
        help="context for retrieval",
    )
    parser.add_argument(
        "--load_path_q",
        default="./encoder/q_encoder_1",
        type=str,
        help="q_encoder saved path",
    )

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    q_encoder = BertEncoder.from_pretrained(args.load_path_q).cuda()

    # org_dataset = load_from_disk(args.dataset_name)
    # train_dataset = org_dataset["train"]
    # validation_dataset = org_dataset["validation"]

    retrieval = RetrievalInference(args, q_encoder, tokenizer, args.context_path)
    retrieval.get_dense_embedding()

    # print("----- val top-5 -----")
    # df = retrieval.retrieval(validation_dataset, topk=5)
    # df = retrieval.get_acc_score(df)
    # retrieval.print_result(df, 5)
    # print("----- val top-10 -----")
    # df = retrieval.retrieval(validation_dataset, topk=10)
    # df = retrieval.get_acc_score(df)
    # retrieval.print_result(df, 10)

    print("----- train top-5 -----")
    df = retrieval.retrieval(None, topk=5)
    df = retrieval.get_acc_score(df)
    retrieval.print_result(df, 5)
    test_query = '서울에 안 유명한 곳 추천해줘!'
    print('----------------------------------------------------------------------------')
    print('Question :', test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '두 시간 정도 시간 떼우기 좋은 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '바다 보면서 혼자 술 마시기 좋은 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '혼자 힐링하기 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '마음이 우울할때 가기 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '친구들과 액티비티 하기 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '부모님과 함께 가기 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '여자친구와 함께 가서 놀기 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '서울에 놀이똥산 같은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '아이들과 과학적 체험이 가능한곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '궁궐 보기 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '생명의 소중함이 느껴지는곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '자존감 회복에 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '생각 비우기 좋은 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '술 마시면서 놀기 좋은곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '강남 빼고 갈만한곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '해돋이가 아름다운곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '시원한곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '물 맑은 계곡'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '실연당한 마음을 보듬어줄 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '스피드를 온몸으로 느끼는 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')   
    test_query = '밥 맛있고 물가 싸고 쾌적한 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '노상방뇨가 가능한 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '중력이 거꾸로 작용하는 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '유동 인구가 가장 많은 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')
    test_query = '근처에 산과 계곡이 있는 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '서울에 사람이 별로 없는 벚꽃 구경할 만한 곳'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '가족들과 캠핑할 만한 계곡 주변 장소'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '수원시 주변에 여자친구와 갈 만한 미술관'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')    
    test_query = '금속탐지기와 취미생활이 가능한 장소'
    print('Question :',test_query)
    df = retrieval.retrieval(test_query, topk=5)
    print('----------------------------------------------------------------------------')   
