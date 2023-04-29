import streamlit as st
from model import WaveNet

st.set_page_config(
    page_title="Train", page_icon=None, layout="centered", 
    initial_sidebar_state="auto", menu_items=None)

st.title("Train Your Model")

@st.cache_data
def load_model(n_hidden, n_embd, dropout, dataset_path = "data/names.txt", device = "cpu"):
    # txt = st.text("loading model")
    model = WaveNet(
        out_dir = "pages/temp/model.pth", 
        hparameters = {
            "n_hidden":n_hidden, 
            "dropout":dropout,
            "n_embd" : n_embd
        }, 
        dataset_path = dataset_path, 
        device = device
    )
    # txt.write("done")
    
    return model


st.markdown("""
This page allows you to build and train your own model.
To achieve this :

- Set the hyperparameters of your model and click build model
- Choose the training details like number of iterations and batch size below
- Click "Start Training"
- After the model has finished training, click download checkpoint to download your trained model

Note: This checkpoint that you download can later be used to generate names in the "Generate" section

---
""")

file = st.file_uploader("Upload a dataset")
if file:
    uploaded_data = file.getvalue()
    # st.write(uploaded_data)
    with open("pages/temp/dataset.txt", "wb") as destination:
        destination.write(uploaded_data)

with st.form("f1"):
    st.markdown("""
    ### Model Architecture
    """)
    cols = st.columns(2)

    with cols[0]:
        n_hidden = st.number_input("Number Of Hidden Units", min_value = 32, max_value = 256, step = 1, value = 128)
        n_embd = st.number_input("Embedding Dimensions", min_value = 2, max_value = 30, value = 24)


    with cols[1]:
        dropout = st.number_input("Enter Value Of Dropout", min_value = 0.0, max_value = 1.0, value = 0.5)
    
    submitted = st.form_submit_button("Build Model")
    
    if submitted:
        if file is None:
            st.error("No dataset provided")
            st.stop()
        model = load_model(n_hidden, n_embd, dropout, dataset_path = "pages/temp/dataset.txt")
        st.session_state.model = model
        st.markdown(f"""
        ###### Model Successfully Built, parameter count: {model.parameter_count}
        """)
    
trained = False
st.empty()
st.empty()
with st.form("f2"):
    
    st.markdown("""
    ### Training Details
    """)
    cols = st.columns(2)

    with cols[0]:
        iterations = st.number_input("Number Of Training Iters", min_value = 1000, max_value = 300000, step = 100, value = 200000)
        track_every = st.number_input("Track Every", min_value = 500, max_value = 20000, step = 100, value = 2500)

    with cols[1]:
        batch_size = st.number_input("Batch Size", min_value = 2, max_value = 64, step = 1, value = 32)
    
    submitted = st.form_submit_button("Start Training")
    
    if submitted:
        if file is None:
            st.error("No dataset provided")
            st.stop()
            
        if "model" not in st.session_state:
            st.error("Model not yet built")
            st.stop()
        chart = st.line_chart()
        st.session_state.model.train(iterations, batch_size=batch_size, chart = chart, track_every=track_every)
        
        trained = True

if trained:
    checkpoint = open("pages/temp/model.pth","rb")

    db = st.download_button(label = "Download Model Checkpoint",
                            file_name = "checkpoint.pth",
                            data = checkpoint)