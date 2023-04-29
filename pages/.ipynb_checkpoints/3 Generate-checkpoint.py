import streamlit as st
from model import *
st.set_page_config(
    page_title="Generator", page_icon=None, layout="centered", 
    initial_sidebar_state="auto", menu_items=None)

st.title("Generator")

def load_model(path):
    model = WaveNet(
        pre_trained = True,
        device = 'cpu',
        checkpoint = path
    )
    
    st.session_state.model_2 = model

@st.cache_data
def convert_to_string(names:list):
    return "\n".join(names)

if "model_2" not in st.session_state:
    load_model("checkpoints/model_names2.pth")
    st.session_state.model_type = "Default"
    
    

tab1, tab2 = st.tabs(["Use Pretrained Model", "Upload your own checkpoint"])

with tab1:
    st.markdown('''This version of our model was trained on 32K popular english baby names from [ssa.gov](https://www.ssa.gov/oact/babynames/), as a result, this model is capable of generating new common-sounding baby names''')

    st.markdown('''
    ---

    ### Model Stats:''')
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label = "Parameters", value = "93,923")
        st.metric(label = "Validation Loss", value = "1.997")

    with col2:
        st.metric(label = "Iterations trained", value = "200,000")
        st.metric(label = "Test Loss", value = "1.995")

    st.markdown('''---''')
    
    form = st.form("f1")
    db = st.empty()
    form.write("Name Generator")
    name_count = form.slider("Number of Names To Generate", min_value = 1)
    submitted = form.form_submit_button("Generate")
    
    
    if submitted:
        if st.session_state.model_type == "Custom":
            load_model("checkpoints/model_names2.pth")
            st.session_state.model_type = "Default"
        
        names = st.session_state.model_2.sample(name_count)
        
        st.dataframe(names, height = min(400, name_count*41), use_container_width =True)
        db.download_button(label="download as plain text", 
                           data = convert_to_string(names), 
                           file_name = "names_generated.txt")



with tab2:
    st.markdown('''
    Here you can upload a checkpoint of your own model. (Hint : You can train your own model by clicking on the train page in the panel on the left)
    
    ---''')
    upload = st.file_uploader("upload your own model checkpoint")
    check = st.button("load model")
    if check:
        if upload is None:
            st.error("No checkpoint uploaded")
            st.stop()
        with open("test.pth", "wb") as file:
            uploaded_data = upload.getvalue()
            file.write(uploaded_data)

        load_model("test.pth")
        st.session_state.model_type = "Custom"
        
    st.markdown("---")
    
    form = st.form("f2")
    db = st.empty()
    form.write("Generator")
    name_count = form.slider("Number of items To Generate", min_value = 1)
    
    submitted = form.form_submit_button("Generate")
    
    if submitted:
        names = st.session_state.model_2.sample(name_count)
        
        st.dataframe(names, height = min(400, name_count*41), use_container_width =True)
        db.download_button(label="download as plain text", 
                           data = convert_to_string(names), 
                           file_name = "items_generated.txt")

