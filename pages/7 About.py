import streamlit as st

st.set_page_config(
    page_title="About", page_icon=None, layout="centered", 
    initial_sidebar_state="auto", menu_items=None)

st.markdown('''
### What is a language model?🤔

A language model is exactly what it sounds like:- it is a machine learning model trained on natural language data (English in this case), that aims to complete the sequence provided in the input by trying to imitate the sequences already shown in the data set.

For example : If a sequence "The dog was" is input, the model may predict the appropriate output as "cute" or "happy" or "excited".

Ultimately, the language model's is nothing but a sequence/sentence completer.

<br>

### Wait... is this like chat-GPT? 😲

YES, in many ways. GPT-3.5, the underlying language model that powers chat-GPT is also a language model, albeit a very powerful one with more than [175B parameters](https://lifearchitect.ai/chatgpt/).

(Ours is a bit more modest with <500,000 parameters 😅)
![img](https://thumbs.gfycat.com/EllipticalCostlyChrysomelid-size_restricted.gif)
<br>

### IS AI GOING TO TAKE OVER THE WORLD LIKE SKYNET!!?? 😱
NO! At least not the language models that we have .
And this is exactly what this project aims to address!

Contrary to the overhyped media headlines, Language Models are nothing but sentence completion engines. THEY HAVE NO SENTIMENT, even when they seem to be, they are just hallucinating their training data.

They simply cannot control what they cannot access.


Here's what we *should* be concerned about instead (rather than an AI invasion or jobs going away):-

- **Mass Disinformation**:
    - Since the LLMs are trained over the internet's data, they are excellent at mimicking human phrases, as a result we should be weary of the consequences if let's say an ill intended party spreads heaps of LLM generated content over the internet. It can not only feel genuine, but may even cause **panic, fear or worse, mass disinformation** about some topic
    

### About the creator:

This web app is created by Jayant Taneja, presently a computer science and engineering major from India
''', unsafe_allow_html = True)