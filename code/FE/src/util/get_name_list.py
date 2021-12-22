import os
import json
import streamlit as st
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


@st.cache(allow_output_mutation=True)
def get_name_list():
    data_path = os.getenv("DATA_PATH")
    with open(f"{data_path}Sparse/pair.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    name_list = [""]
    for area in data:
        for i, v in enumerate(data[area]["관광지"]):
            if i == 0:
                continue
            name_list.append(v)
    name_list.sort()

    return name_list
