import torch
import os
import random
import json
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict, Value, Features, load_from_disk
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from utils import retrieve_from_embedding

def TrainInbatchDatasetJson(tokenizer, dataset: Dataset, max_length:int) -> TensorDataset:
    print("get in-batch training sample")

    p_seqs = tokenizer(
        dataset["context"], max_length=max_length ,padding="max_length", truncation=True, return_tensors="pt"
    )
    q_seqs = tokenizer(
        dataset["query"], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    return TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )

def TrainRetrievalDataset(tokenizer_name: str, dataset_name: str) -> TensorDataset:
    print("get in-batch training sample")
    org_dataset = load_from_disk(dataset_name)
    train_data = org_dataset["train"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    question = train_data["question"]
    context = train_data["context"]

    p_seqs = tokenizer(
        context, padding="max_length", truncation=True, return_tensors="pt"
    )
    q_seqs = tokenizer(
        question, padding="max_length", truncation=True, return_tensors="pt"
    )

    return TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )
