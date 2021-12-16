from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from dense import DenseRetrieval
import numpy as np

import uvicorn
import sys

sys.path.append("/opt/ml/final-project-level3-nlp-11/code/MODEL/sparse")
from elastic_search import ElasticSearch

sys.path.append("/opt/ml/final-project-level3-nlp-11/sparse")
from retrieval_similar import SimilarSparse

class ResItem(BaseModel):
    location: str
    places: list

app = FastAPI()

### 장소 묘사
p_model = "./saved_models/p_encoder"
q_model = "./saved_models/q_encoder"
tokenizer = "klue/bert-base"
embedding_path = "./saved_models/dense_embedding.bin"

model : DenseRetrieval = None

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
    model = DenseRetrieval(p_encoder_model=p_model, q_encoder_model=q_model, tokenizer_name=tokenizer, pickle_path=embedding_path, token_length=128, es=es)
    similar_retrieval_model = SimilarSparse(pre_tokenizer=tokenizer, data_path=data_path, json_name=json_file_name)
    similar_retrieval_model.get_sparse_embedding()
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
def get_similar(query:str=None, location:str="전국"):
    true_false = similar_retrieval_model.get_query_contexts(query)
    if not true_false:
        print("No Matches, Run dense with query")
        exit()
    else:
        result_contents = similar_retrieval_model.retrieve(similar_retrieval_model.query_contexts, 5, location)
        res = {"location": location, "places": list(result_contents)}
        result = ResItem(**res)
        res_json = jsonable_encoder(result)
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
def predict(query:str=None, location:str="전국"):
    """
        장소묘사 Query에 대한 predict
        API : {IP_path}/prediciton/?query={query}&location={location}
    """
    pred = model.inference(single_query=query, area=location, use_elastic=True)
    places = get_dense_data(pred)
    res = {"location": location, "places": list(places)}
    result = ResItem(**res)
    res_json = jsonable_encoder(result)
    return JSONResponse(content=res_json)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)