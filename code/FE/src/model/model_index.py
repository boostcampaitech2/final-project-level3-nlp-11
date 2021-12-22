import streamlit as st
import os
import numpy as np
from model.retrieval_similar import SimilarSparse
from api.tour_api import Api
from model.elastic_search import ElasticSearch
from model.dense import DenseRetrieval
from dotenv import load_dotenv


env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def get_data(retriever, sentence, area):

    true_false = retriever.get_query_contexts(sentence)
    if not true_false:
        result_contents = ""
    else:
        result_contents = retriever.retrieve(retriever.query_contexts, 5, area)
        result_contents = Api().get_tour_topK(result_contents, area)

    return result_contents


def get_dense_data(df, area):
    # TODO 스코어를 더해서 계산할지 아니면 그냥 중복 제거만 할지 고민
    name_list = df["place"].tolist()

    name_set = set(name_list)

    score_info = []

    for name in name_set:

        score_info.append([name, df[df["place"] == name]["scores"].sum()])

    score_info = sorted(score_info, key=lambda x: x[1], reverse=True)
    score_info = np.array(score_info)

    result = Api().get_tour_topK(score_info[:5, 0], area)

    return result


@st.cache(allow_output_mutation=True)
def get_model(data_path):
    retriever = SimilarSparse(
        "klue/bert-base",
        data_path=f"{data_path}Sparse",
        json_name="pair.json",
    )

    retriever.get_sparse_embedding()

    return retriever


@st.cache(allow_output_mutation=True)
def get_dense():
    data_path = os.getenv("DATA_PATH")

    p_encoder_path = f"{data_path}Dense/p_encoder"
    q_encoder_path = f"{data_path}Dense/q_encoder"
    tokenizer_name = "monologg/kobigbird-bert-base"
    pickle_path = f"{data_path}Dense"
    token_length = 1024
    es = ElasticSearch()

    dense_model = DenseRetrieval(
        p_encoder_path, q_encoder_path, tokenizer_name, pickle_path, token_length, es
    )
    dense_model.get_embedding()
    return dense_model
