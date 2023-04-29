import streamlit as st

st.set_page_config(
    page_title="Home", page_icon=None, layout="centered", 
    initial_sidebar_state="auto", menu_items=None)


st.title("Generative Model Playground")

st.markdown('''

Welcome to Generative Model Playground !

This is an interactive web app that aims to showcase an experimental character-level language model that aims to generate natural language sequences similar to those found in the dataset.

Go ahead and click on the sidebar👈 to explore more !

![img](https://i.gifer.com/origin/dc/dc412623146610157eb73e727f4d16bc.gif)
<br>



''', unsafe_allow_html = True)