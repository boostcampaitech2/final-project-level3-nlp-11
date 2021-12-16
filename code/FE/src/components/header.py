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
    width ="50vw" 
    style="
        float:left;
        margin-right:5px;
        margin-top:1px;
        vertical-align:middle
    " """

    home_text_style = """
    style="
        float:left;
        font-size:38px;
        font-weight:bold
    " """

    home_text = "저기 어때"
    icon_src = '''src="data:image/png;base64,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"'''

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
