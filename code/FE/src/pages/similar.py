import os

import streamlit as st

from styles.main_list import main_list_style
from components.main_list_item import parser_data
from model.model_index import get_data, get_model
from util.get_name_list import get_name_list
from util.theme_name_tuple import get_teme_name_tuple
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def similar_page():

    root_path = os.getenv("ROOT_PATH")
    col1, col2 = st.columns([1, 7])
    # TODO 전국 추가하기 (get_teme_name_tuple)
    with col1:
        area = st.selectbox(
            "지역",
            get_teme_name_tuple(),
        )
    nametuple = get_name_list()
    with col2:
        sentence = st.selectbox("유사 명소 검색", nametuple)
    result_contents = ""

    if sentence != "" and area:
        with st.spinner("찾는 중!..."):
            data_path = os.getenv("DATA_PATH")
            retriever = get_model(data_path)
            result_contents = get_data(retriever, sentence, area)
            if not result_contents:
                st.write("찾기 실패!")
    link = f"""
            {main_list_style()}
            <hr>
            <div style="font-size:62.5%">
                {parser_data(result_contents,area)}
            </div>         
            """
    st.markdown(link, unsafe_allow_html=True)
