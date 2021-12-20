import streamlit as st

import io
import os
import yaml

from PIL import Image
from styles.main_list import main_list_style
from components.main_list_item import parser_data
from util.theme_name_tuple import get_teme_name_tuple
from api.model import ModelApi
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
            data = ModelApi().request_predict(sentence, area)

            if not data:
                st.write("한국어 또는 충분히 긴 문장으로 작성해주시기 바랍니다.")

    link = f"""
            {main_list_style()}
            <hr>
            <div style="font-size:62.5%">
                {parser_data(data,area,sentence)}
            </div>         
            """
    st.markdown(link, unsafe_allow_html=True)
