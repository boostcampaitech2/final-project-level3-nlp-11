from typing import Optional
from datetime import datetime
from pytz import timezone
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from torch._C import BenchmarkConfig
from dense import DenseRetrieval
import numpy as np
import os
from dotenv import load_dotenv
from logger import Logger

import uvicorn
import sys

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)
from google.auth.environment_vars import CREDENTIALS

sys.path.append("/opt/ml/final-project-level3-nlp-11/code/MODEL/sparse")
from elastic_search import ElasticSearch

sys.path.append("/opt/ml/final-project-level3-nlp-11/sparse")
from retrieval_similar import SimilarSparse


class ResItem(BaseModel):
    location: str
    places: list


class SurveyItemIn(BaseModel):
    query: str
    location: str
    place: str
    is_good: bool


app = FastAPI()
search_logger = Logger(
    table_id="ai-esg-trip-recommendation.log_data.search_result",
    credential_json_path=os.environ.get(CREDENTIALS),
)
similar_logger = Logger(
    table_id="ai-esg-trip-recommendation.log_data.log_similar_data",
    credential_json_path=os.environ.get(CREDENTIALS),
)
survey_logger = Logger(
    table_id="ai-esg-trip-recommendation.log_data.log_survey_data",
    credential_json_path=os.environ.get(CREDENTIALS),
)

### 장소 묘사
p_model = "./saved_models/p_encoder"
q_model = "./saved_models/q_encoder"
tokenizer = "klue/bert-base"
embedding_path = "./saved_models/dense_embedding.bin"

model: DenseRetrieval = None

### 유사명소
data_path = "../../data"
json_file_name = "pair.json"
tokenizer = "klue/bert-base"

similar_retrieval_model: SimilarSparse = None


### 유사명소
data_path = "../../data"
json_file_name = "pair.json"
tokenizer = "klue/bert-base"

similar_retrieval_model: SimilarSparse = None


@app.on_event("startup")
def server_on_event():
    # model load
    global model
    global similar_retrieval_model
    es = ElasticSearch()
    model = DenseRetrieval(
        p_encoder_model=p_model,
        q_encoder_model=q_model,
        tokenizer_name=tokenizer,
        pickle_path=embedding_path,
        token_length=128,
        es=es,
    )
    similar_retrieval_model = SimilarSparse(
        pre_tokenizer=tokenizer, data_path=data_path, json_name=json_file_name
    )
    similar_retrieval_model.get_sparse_embedding()

    ### for predicting faster
    locations = [
        "전국",
        "서울",
        "인천",
        "대구",
        "부산",
        "경기도",
        "강원도",
        "충청남도",
        "충청북도",
        "경상북도",
        "경상남도",
        "전라북도",
        "전라남도",
        "제주도",
    ]
    for location in locations:
        pred = model.inference(single_query="테스트", area=location, use_elastic=True)
    print("Model loaded !!")


@app.on_event("shutdown")
def server_off_event():
    print("The Server is Closed !!")


@app.get("/")
def main_page():
    print("main page!!")
    return {"this is ": "main page"}


###유사 명소
@app.get("/get_similar/")
async def get_similar(
    background_task: BackgroundTasks, query: str = None, location: str = "전국"
):
    true_false = similar_retrieval_model.get_query_contexts(query)
    if not true_false:
        print("No Matches, Run dense with query")
        exit()
    else:
        result_contents = similar_retrieval_model.retrieve(
            similar_retrieval_model.query_contexts, 5, location
        )
        res = {"location": location, "places": list(result_contents)}
        result = ResItem(**res)
        res_json = jsonable_encoder(result)
        background_task.add_task(log_similar, query, location, res_json)
        return JSONResponse(content=res_json)


### 장소 묘사 predict
def get_dense_data(df):
    # TODO 스코어를 더해서 계산할지 아니면 그냥 중복 제거만 할지 고민
    name_list = df["place"].tolist()
    name_set = set(name_list)
    score_info = []
    for name in name_set:
        score_info.append([name, df[df["place"] == name]["scores"].sum()])
    score_info = sorted(score_info, key=lambda x: x[1], reverse=True)
    score_info = np.array(score_info)
    return score_info[:5, 0]


@app.get("/predict/")
async def predict(
    background_task: BackgroundTasks, query: str = None, location: str = "전국"
):
    """
        장소묘사 Query에 대한 predict
        API : {IP_path}/predici/?query={query}&location={location}
    """
    pred = model.inference(single_query=query, area=location, use_elastic=True)
    places = get_dense_data(pred)
    res = {"location": location, "places": list(places)}
    result = ResItem(**res)
    res_json = jsonable_encoder(result)
    background_task.add_task(log_search, query, location, res_json)
    return JSONResponse(content=res_json)


@app.post("/survey/")
async def survey(background_task: BackgroundTasks, item: SurveyItemIn):
    now_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    background_task.add_task(
        log_survey, item.query, item.location, item.place, now_time, item.is_good
    )
    return {"timestamp": now_time}


def log_search(query: str, location: str, res_data: dict) -> None:
    result = []
    now_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    for place in res_data["places"]:
        row = {
            "query": query,
            "location": location,
            "place": place,
            "time": now_time,
        }
        result.append(row)
    search_logger.insert_log(result)


def log_similar(query: str, location: str, res_data: dict) -> None:
    result = []
    now_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    for place in res_data["places"]:
        row = {
            "location": location,
            "input_place": query,
            "output_place": place,
            "time": now_time,
        }
        result.append(row)
    similar_logger.insert_log(result)


def log_survey(
    query: str, location: str, place: str, now_time: datetime, is_good: bool
) -> None:
    row = {
        "query": query,
        "location": location,
        "place": place,
        "time": now_time,
        "is_good": is_good,
    }
    survey_logger.insert_log([row])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
