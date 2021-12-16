from fastapi import FastAPI
import uvicorn
import sys
sys.path.append("/opt/ml/final-project-level3-nlp-11/code/MODEL/sparse")
from elastic_search import ElasticSearch
from dense import DenseRetrieval


app = FastAPI()
p_model = "./saved_models/p_encoder"
q_model = "./saved_models/q_encoder"
tokenizer = "klue/bert-base"
embedding_path = "./saved_models/dense_embedding.bin"

model : DenseRetrieval = None

@app.on_event("startup")
def server_on_event():
    # model load
    global model
    es = ElasticSearch()
    model = DenseRetrieval(p_encoder_model=p_model, q_encoder_model=q_model, tokenizer_name=tokenizer, pickle_path=embedding_path, token_length=128, es=es)
    print("Model loaded !!")

@app.on_event("shutdown")
def server_off_event():
    print("The Server is Closed !!")

@app.get("/")
def main_page():
    print("main page!!")
    return {"this is ": "main page"}

# log


#유사 명소
# 장소묘사
# path/prediciton/?query={query}&location={location}
@app.get("/prediction/")
def predict(query:str=None):
    pred = model.inference(single_query=query, use_elastic=True)
    print(pred)
    result = {}
    return {"result": "ok"} # place이름 list

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)