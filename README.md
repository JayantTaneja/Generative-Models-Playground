# Generative-Models-Playground

This is an interactive web app that aims to showcase an experimental character-level language model that aims to generate natural language sequences similar to those found in the dataset.

## Use:

### Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jayanttaneja-generative-models-playground.streamlit.app/)


### Install Locally 

```bash
>>> git clone https://github.com/JayantTaneja/Generative-Models-Playground.git

>>> cd Generative-Models-Playground
>>> pip install streamlit
>>> pip install -r requirements.txt
>>> streamlit run Home.py
```

---

## Features

- ### Train Your Own Character Level Language Model
    - Visit [Training Your Own Model](https://jayanttaneja-generative-models-playground.streamlit.app/Train)
    - Upload a text dataset (each line should contain only one (1) single training example)
    - Choose Your Hyperparameters and Training Details
    - Click "Start Training"
    - Once the training is complete, download your model's checkpoint

- ### Generate Novel Samples
    - Visit [Generating Your Own Samples](https://jayanttaneja-generative-models-playground.streamlit.app/Generate)
    - Use your own checkpoint 
    - Or, alternatively, Use a pretrained checkpoint that generates baby names

- ### Explore the process of generating a new sample Token-By-Token
    - Visit [How Does It Work](https://jayanttaneja-generative-models-playground.streamlit.app/How_Does_It_Work)
    - Enter a start context
    - Click "Generate"
    - Click "Next Step" until a complete name is generated

- ### Check Your English Skills
    - Visit [English Or Not English?](https://jayanttaneja-generative-models-playground.streamlit.app/English_Or_Not_English)
    - Play against a mix of random AI generated words and real English words to guess which is which 


## Model Architecture
This model is a modification of the [WaveNet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio) Model Architecture

![Model Architecture GIF](https://assets-global.website-files.com/621e749a546b7592125f38ed/62227b1d1dd26da452c9e160_unnamed-2.gif)

For more details check out [Model Details](https://jayanttaneja-generative-models-playground.streamlit.app/Model_Details)