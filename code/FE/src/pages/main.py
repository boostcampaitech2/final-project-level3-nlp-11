import streamlit as st

import io
import os
import yaml

from PIL import Image
from styles.main_list import main_list_style
from components.main_list_item import parser_data
from util.theme_name_tuple import get_teme_name_tuple
from model.model_index import get_dense, get_dense_data
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def main_page():
    root_path = os.getenv("ROOT_PATH")

    col1, col2 = st.columns([1, 9])
    with col2:
        sentence = st.text_input("명소 검색")
    with col1:
        area = st.selectbox(
            "지역",
            get_teme_name_tuple(),
        )
    data = ""
    if sentence and area:
        with st.spinner("찾는 중!..."):
            dense_model = get_dense()
            df = dense_model.inference(sentence, 25, area)
            if df is None:
                st.write("한국어가 아니거나 너무 짧은 문장입니다")
            else:
                data = get_dense_data(df, area)

    link = f"""
            {main_list_style()}
            <hr>
            <div style="font-size:62.5%">
                {parser_data(data,area)}
            </div>         
            """
    st.markdown(link, unsafe_allow_html=True)
