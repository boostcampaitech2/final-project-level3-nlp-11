import pickle
import torch
import numpy as np
import pandas as pd
import os
import json
import sklearn

from datasets import load_from_disk
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, Value, Features

def retrieve_from_embedding(
    dataset_path: str,
    q_encoder: AutoModel,
    tokenizer_name: str,
    emb_path: str,
    topk: int,
    context_path: str,
) -> pd.DataFrame:

    """
    Input : Dataset path(question), Question encoder, Tokenizer, Embedding path, number of top score documents (topk), wikipedia json file path
    Output : Dataframe of topk dense retrieval documents
    """

    ## get relevant doc bulk
    dataset = load_from_disk(dataset_path)
    # dataset = dataset['validation']
    dataset = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(emb_path, "rb") as file:
        p_embedding = pickle.load(file)
    q_embs = []

    with torch.no_grad():
        q_encoder.eval()

        for q in tqdm(dataset["question"]):
            q = tokenizer(
                q, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = q_encoder(**q).pooler_output.to("cpu").detach().numpy()
            q_embs.append(q_emb)

    q_embs = torch.Tensor(q_embs).squeeze()
    result = torch.matmul(q_embs, torch.transpose(p_embedding, 0, 1))

    if not isinstance(result, np.ndarray):
        result = np.array(result.tolist())
    doc_scores = []
    doc_indices = []
    for i in range(result.shape[0]):
        sorted_result = np.argsort(result[i, :])[::-1]
        doc_scores.append(result[i, :][sorted_result].tolist()[:topk])
        doc_indices.append(sorted_result.tolist()[:topk])
    ## get relevant doc bulk

    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    total = []
    for idx, example in enumerate(tqdm(dataset, desc="Dense retrieval: ")):
        context_array = []
        for pid in doc_indices[idx]:
            context = "".join(contexts[pid])
            context_array.append(context)
        tmp = {
            # Query와 해당 id를 반환합니다.
            "question": example["question"],
            "id": example["id"],
            # Retrieve한 Passage의 id, context를 반환합니다.
            "context_id": doc_indices[idx],
            "context": context_array,
            "scores": doc_scores[idx],
        }
        if "context" in example.keys() and "answers" in example.keys():
            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
            tmp["original_context"] = example["context"]
            tmp["answers"] = example["answers"]
        total.append(tmp)

    cqas = pd.DataFrame(total)
    return cqas


def retrieval_acc(df: pd.DataFrame, topk: int) -> float:
    df["correct"] = False
    df["correct_rank"] = 0
    for i in tqdm(range(len(df)), desc="check tok_n"):
        count = 0
        for context in df.iloc[i]["context"]:
            count += 1
            if df.iloc[i]["original_context"] == context:
                df.at[i, "correct"] = True
                df.at[i, "correct_rank"] = count

    accuracy = (df["correct"].sum() / len(df),)

    return accuracy


def inbatch_input(batch: list, batch_size: int, device: str) -> torch.Tensor:
    targets = torch.arange(batch_size).long()
    targets = targets.to(device)

    p_inputs = {
        "input_ids": batch[0].to(device),
        "attention_mask": batch[1].to(device),
        "token_type_ids": batch[2].to(device),
    }
    q_inputs = {
        "input_ids": batch[3].to(device),
        "attention_mask": batch[4].to(device),
        "token_type_ids": batch[5].to(device),
    }

    return q_inputs, p_inputs, targets


def inbatch_sim_scores(
    q_outputs: torch.Tensor, p_outputs: torch.Tensor
) -> torch.Tensor:
    sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
    sim_scores = F.log_softmax(sim_scores, dim=1)

    return sim_scores


def neg_sample_input(
    batch: list, batch_size: int, device: str, num_neg: int
) -> torch.Tensor:
    targets = torch.zeros(batch_size).long()
    targets = targets.to(device)
    p_inputs = {
        "input_ids": batch[0].view(batch_size * (num_neg + 1), -1).to(device),
        "attention_mask": batch[1].view(batch_size * (num_neg + 1), -1).to(device),
        "token_type_ids": batch[2].view(batch_size * (num_neg + 1), -1).to(device),
    }

    q_inputs = {
        "input_ids": batch[3].view(batch_size, -1).to(device),
        "attention_mask": batch[4].view(batch_size, -1).to(device),
        "token_type_ids": batch[5].view(batch_size, -1).to(device),
    }
    return q_inputs, p_inputs, targets


def neg_sample_sim_scores(
    q_outputs: torch.Tensor, p_outputs: torch.Tensor, batch_size: int, num_neg: int
) -> torch.Tensor:
    p_outputs = torch.transpose(p_outputs.view(batch_size, num_neg + 1, -1), 1, 2)

    q_outputs = q_outputs.view(batch_size, 1, -1)
    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
    sim_scores = sim_scores.view(batch_size, -1)

    sim_scores = F.log_softmax(sim_scores, dim=1)

    return sim_scores

def get_train_dataset(data_path:str="/opt/ml/final-project-level3-nlp-11/data/MODEL"):
    with open(os.path.join(data_path, "pair.json"), "r", encoding="utf-8-sig") as f:
        pair_data = json.load(f)
    
    with open(os.path.join(data_path, "query_set.json"), "r", encoding="utf-8-sig") as f:
        val_query_data = json.load(f)
    
    context_info_dict = []
    places = []
    contexts = []
    query_dict = []
    
    for area in pair_data:
        for place in pair_data[area]['관광지']:
            for query in val_query_data[area]['관광지'][place]:
                query_dict.append({"query": query, "place": place})
            for pair in pair_data[area]['관광지'][place]:
                context_info_dict.append(
                    {
                        "area":area,
                        "place":place,
                        "query":pair["query"],
                        "context":pair["context"],
                    }
                )
                places.append(place)
                contexts.append(pair["context"])
    
    df_cont_info = sklearn.utils.shuffle(pd.DataFrame(context_info_dict), random_state=2022)
    df_query_info = pd.DataFrame(query_dict)
    
    #info_train, info_val = train_test_split(
    #    df_info, test_size=0.2, random_state=2022, stratify=df_info["place"])
    
    f_c = Features(
        {
            "area": Value(dtype="string", id=None),
            "place": Value(dtype="string", id=None),
            "query": Value(dtype="string", id=None),
            "context": Value(dtype="string", id=None),
        }
    )
    
    f_v = Features(
        {
            "place": Value(dtype="string", id=None),
            "query": Value(dtype="string", id=None),
        }
    )
    return DatasetDict({"train": Dataset.from_pandas(df_cont_info, features=f_c),
                        "validation": Dataset.from_pandas(df_query_info, features=f_v)}), contexts, places