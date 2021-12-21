import sys
import os
import pickle
import argparse
import json
import wandb
from time import time

from typing import List, Tuple, NoReturn, Any, Optional, Union
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    set_seed,
)

from dataset import *
from utils import *


class DenseRetrieval:
    def __init__(
        self,
        args,
        p_encoder: AutoModel,
        q_encoder: AutoModel,
    ) -> None:
        self.args = args
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name)
        
        self.p_embedding = None
        self.train_datadict, self.contexts, self.places = get_train_dataset(self.args.dataset_path)
        torch.cuda.empty_cache()
    
    def get_embedding(self):
        emd_path = self.args.save_pickle_path

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
                    p, padding="max_length", max_length=self.args.token_length, truncation=True, return_tensors="pt"
                ).to("cuda")
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

    def get_relevant_doc(self, query: str, k = 1):
        q_encoder = self.q_encoder
        q_embs = []

        with torch.no_grad():
            q_encoder.eval()
            print('getting Query X Passage scores')
            q = tokenizer(
                query, max_length=self.args.token_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = q_encoder(**q).pooler_output.to("cpu").detach().numpy()

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
    
    def train(self):
        print("="*30, " Start Train !!! ", "="*30)
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        args = self.args
        batch_size = args.batch_size
        
        train_data = self.train_datadict["train"]
        val_data = self.train_datadict["validation"]
        tokenizer = self.tokenizer

        # get in-batch dataset
        train_dataset = TrainInbatchDatasetJson(
            tokenizer=self.tokenizer,
            dataset=train_data,
            max_length=self.args.token_length
        )
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
        )
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
        )

        global_step = 0
        epoch = 0
        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()
        best_top = {1:0, 5:0, 10:0, 30:0, 50:0, 100:0}

        for _ in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):
            print("-"*20, f" epoch {epoch} ", "-"*20)
            epoch += 1
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()

                    global_step += 1

                    q_inputs, p_inputs, targets = inbatch_input(
                        batch, batch_size, self.device
                    )

                    p_outputs = p_encoder(**p_inputs).pooler_output
                    q_outputs = q_encoder(**q_inputs).pooler_output

                    sim_scores = inbatch_sim_scores(q_outputs, p_outputs)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    p_encoder.zero_grad()
                    q_encoder.zero_grad()

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

                    if global_step % args.log_step == 0:
                        wandb.log({"loss": loss}, step=global_step)
            
            print(f'train epoch : {epoch} done')
            if epoch > args.val_start_epoch:
                with torch.no_grad():
                    print(" testing for validation set ", "-"*40)
                    p_encoder.eval()

                    p_embs = []
                    for p in tqdm(self.contexts):
                        p = tokenizer(p, max_length=self.args.token_length, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                        p_emb = p_encoder(**p).pooler_output.to('cpu').numpy()
                        p_embs.append(p_emb)

                    p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

                    top = {1:0, 5:0, 10:0, 30:0, 50:0, 100:0}
                    print("tmp embdding done")
                    q_encoder.eval()
                    for datum in tqdm(val_data):
                        query = datum["query"]
                        sample_place = datum["place"]

                        q_seqs_val = tokenizer([query], max_length=self.args.token_length, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                        q_emb = q_encoder(**q_seqs_val).pooler_output.to('cpu')  #(num_query, emb_dim)

                        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
                        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

                        check = False
                        for idx in rank[:100]:
                            if sample_place == self.places[idx]:
                                for k in top:
                                    if idx < k:
                                        top[k] += 1
                                        if k == 100:
                                            check = True
                                if check:
                                    break
                print("-"*20," validation set accuracy ", "-"*20)
                for k in top:
                    top[k] /= len(val_data)
                    print(f"for top-{k} acc : {top[k]*100:0.4f}")
                    wandb.log({f"val/top-{k} acc": top[k]}, step=epoch)
                for k in top:
                    if top[k] > best_top[k]:
                        best_top[k] = top[k]
                p_encoder.save_pretrained(
                    save_directory=f"{args.save_path_p}/epoch{epoch}"
                )
                q_encoder.save_pretrained(
                    save_directory=f"{args.save_path_q}/epoch{epoch}"
                )
        print("all done!!", '-'*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--save_path_q", default="./encoder/q_encoder", type=str, help=""
    )
    parser.add_argument(
        "--save_path_p", default="./encoder/p_encoder", type=str, help=""
    )
    parser.add_argument(
        "--dataset_path", default="/opt/ml/final-project-level3-nlp-11/data/MODEL", type=str, help=""
    )
    parser.add_argument(
        "--tokenizer_name",
        default="monologg/kobigbird-bert-base",
        type=str,
        help="",
    )
    parser.add_argument(
        "--token_length",
        default=1024,
        type=int,
        help="",
    )

    parser.add_argument(
        "--run_name", default="dense_retrieval", type=str, help="wandb run name"
    )
    parser.add_argument(
        "--num_train_epochs", default=20, type=int, help="number of epochs for train"
    )
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size for train"
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="learning rate for train"
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="weight decay coeff for train"
    )
    parser.add_argument(
        "--random_seed", default=211, type=int, help="random seed for numpy and torch"
    )
    parser.add_argument(
        "--p_enc_name_or_path",
        default="monologg/kobigbird-bert-base",
        type=str,
        help="name or path for p_encoder",
    )
    parser.add_argument(
        "--q_enc_name_or_path",
        default="monologg/kobigbird-bert-base",
        type=str,
        help="name or path for q_encoder",
    )
    parser.add_argument(
        "--save_pickle_path",
        default="/opt/ml/final-project-level3-nlp-11/data/MODEL/dense_embedding.bin",
        type=str,
        help="wiki embedding save path",
    )
    parser.add_argument(
        "--save_epoch", default=10, type=int, help="save encoders per epoch"
    )
    parser.add_argument(
        "--log_step", default=600, type=int, help="log loss to wandb per step"
    )
    parser.add_argument(
        "--val_start_epoch",
        default=10,
        type=int,
        help="validation starting epoch",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="adam epsilon for optimizer"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="gradient accumulation steps for scheduler",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="warmup steps for scheduler"
    )

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    set_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    p_encoder = AutoModel.from_pretrained(args.p_enc_name_or_path).cuda()
    q_encoder = AutoModel.from_pretrained(args.q_enc_name_or_path).cuda()

    wandb.init(entity="ai_esg", name=args.run_name)

    retriever = DenseRetrieval(
        args,
        p_encoder,
        q_encoder,
    )
    retriever.train()
    #retriever.save_embedding(args.save_pickle_path + ".bin")