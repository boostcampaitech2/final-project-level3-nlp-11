import streamlit as st
import extra_streamlit_components as stx


@st.cache(allow_output_mutation=True)
def get_manager():
    return stx.CookieManager()
