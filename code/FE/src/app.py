import streamlit as st
import extra_streamlit_components as stx
import io
import os
import yaml

from pages.info import info_page
from pages.similar import similar_page
from pages.main import main_page

from components.header import header

from PIL import Image
from dotenv import load_dotenv
from util.get_cookie import get_manager

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(
    page_title="저기어때",
    page_icon="https://i.imgur.com/hAy9Ezw.png",
    layout="wide",
    initial_sidebar_state="auto",
)
root_path = os.getenv("ROOT_PATH")


cookie_manager = get_manager()


def main():
    header()
    query_params = st.experimental_get_query_params()

    if not query_params:
        main_page()

    elif query_params["page"][0] == "2":
        info_page(query_params, cookie_manager)

    elif query_params["page"][0] == "3":
        similar_page()
    else:
        main_page()


main()
