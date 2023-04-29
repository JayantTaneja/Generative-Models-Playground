import streamlit as st
from model import *

st.set_page_config(
    page_title="How Does It Work?", page_icon=None, layout="centered", 
    initial_sidebar_state="auto", menu_items=None)

st.title("But How Does It Work?")

st.markdown('''
To start off, first off all, instead of thinking about the model as a giant "Brain", think of it as a sequence completer.

**Like autocomplete ?**

Well, Yes and No. You see, the autocomplete softwares usually just map which "word" pairs are most frequent and simply use that to predit
the next word in the sequence. 

- This model on the other hand uses a probabilistic approach, when it sees the past letters, it predicts the probability of the next letter.

- It then picks randomly (but propotional to their probability or multinomial sampling as we call it) the next letter in the sequence.
- The previous context along with the predicted letter is again fed into the model.

This process is repeated unitl the stop sequence "." special character is predicted.

### Demo

Here you get to see the working in action.

Simply type the start sequence (can be empty) and click generate, the model will generate the word that it thinks will most likely result 
from the starting context and we will show what the model thinks at each step.''')

def load_model(path):
    model = WaveNet(
        pre_trained = True,
        device = 'cpu',
        checkpoint = path
    )
    
    st.session_state.model_2 = model

def show_chart(chart_holder):
    data = pd.DataFrame(
        {
            "Character" : st.session_state.probabilities[st.session_state.show_idx].tolist()[0]
        }, 
        index = letters
    )
    chart_holder.bar_chart(data)
    
    
if "model_2" not in st.session_state:
    load_model("checkpoints/model_names2.pth")
    st.session_state.model_type = "Default"

if "generated" not in st.session_state:
    st.session_state.generated = False
    
letters = ['.'] + [chr(97+i) for i in range(26)]

with st.form("Visualise"):
    start = st.text_input(label = "Enter starting context")    
    submit = st.form_submit_button("Generate")
    
    if submit:
        st.session_state.start = start.lower()
        st.session_state.generated = True
        st.session_state.probabilities, st.session_state.word = st.session_state.model_2.vis_sample(start = st.session_state.start)
        st.session_state.show_idx = 0
        st.session_state.frame_count = len(st.session_state.probabilities)



if st.session_state.generated:
    # txt = st.text(f"{st.session_state.word}")
    # st.text(st.session_state.frame_count)
    # st.text(st.session_state.word)
    text = st.text(f"Word generated so far : {st.session_state.start.title()}")
    if((st.session_state.show_idx + len(st.session_state.start)) >= len(st.session_state.word)):
        next_letter = st.text(f"Next Letter Chosen : <end of sequence>")
    else:
        next_letter = st.text(f"Next Letter Chosen : {st.session_state.word[len(st.session_state.start)+st.session_state.show_idx]}")
    
    
    cols = st.columns(4)
    with cols[3]:
        next_frame = st.button(label = "Next Step")    
    
    with cols[1]:
        prev_frame = st.button(label = "Prev Step")
    
    if "chart" not in st.session_state:
        st.session_state.chart = st.bar_chart()
    show_chart(st.session_state.chart)
    
    if next_frame:
        show_chart(st.session_state.chart)
        text.text(f"Word generated so far : {st.session_state.word[:len(st.session_state.start)+st.session_state.show_idx].title()}")
        st.session_state.show_idx += 1
        if st.session_state.show_idx >= st.session_state.frame_count:
            st.session_state.show_idx -=1
        
    if prev_frame:
        show_chart(st.session_state.chart)        
        text.text(f"Word generated so far : {st.session_state.word[:len(st.session_state.start)+st.session_state.show_idx].title()}")
        st.session_state.show_idx -= 1
        if st.session_state.show_idx <= 0:
            st.session_state.show_idx = 0
        
