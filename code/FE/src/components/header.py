import os
import streamlit as st

from dotenv import load_dotenv


env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def header():
    root_path = os.getenv("ROOT_PATH")

    div_home_style = """
    style="
        text-decoration:none;
        color:rgb(29,51,63)
    " """
    icon_style = """
    width ="80vw" 
    style="
        float:left;
        margin-right:5px;
        margin-top:1px;
        vertical-align:middle
    " """

    home_text_style = """
    style="
        float:left;
        margin-top:7px;
        font-size:38px;
        font-weight:bold
    " """

    home_text = "저기 어때"
    icon_src = '''src="https://i.imgur.com/hAy9Ezw.png"'''

    html = f"""
        <a href="{root_path}" {div_home_style}>
            <img id="title_img" {icon_style} {icon_src}>
            <font id="title_context" {home_text_style}> {home_text} </font>
        </a>
        <a href="{root_path}?page=3"{div_home_style}>
            <font style="float:right;font-size:20px;font-weight:bold;margin-top:15px"> 유사 명소 검색 </font>
        </a>
        <a href="{root_path}" {div_home_style}>
            <font style="float:right;font-size:20px;font-weight:bold;margin-top:15px;margin-right:25px"> 명소 검색 </font>
        </a>
        <br>
        <br>
        <br>

    """
    st.markdown(html, unsafe_allow_html=True)
