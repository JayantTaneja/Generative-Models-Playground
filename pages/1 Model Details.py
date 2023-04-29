import streamlit as st

st.set_page_config(
    page_title="Model Details", page_icon=None, layout="centered", 
    initial_sidebar_state="auto", menu_items=None)

st.title("Model Details")

st.write("This page shows detail of how this language model was built, its architecture, etc.")


st.markdown('''


## Model Architecture
This model is a modification of the [WaveNet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio) Model Architecture

![Model Architecture GIF](https://assets-global.website-files.com/621e749a546b7592125f38ed/62227b1d1dd26da452c9e160_unnamed-2.gif)

<sup>Gif showing how the architecture is made</sup>

The original WaveNet model architecture was bult for reading audio file(hence "wave"), however, our model works on natural language.

Following is how the architecture is laid out:
''', unsafe_allow_html  = True)


st.markdown('''
- **Embedding Layer**
    - This layer provides a look up table to convert our character (or token) into an ```n_embd``` dimensional vector
''')

video1 = open(r"images/Embedding.mp4", "rb")
st.video(video1.read())


st.markdown('''
- Strided ```Conv1d``` Layers
    - These merge 2 tokens into a single output neuron for the next layer. 
''')

video2 = open(r"images/conv.mp4", "rb")
st.video(video2.read())


st.markdown('''
- **Batchnorm 1d**
    - A batch normalization layer that normalizes the data accross the ```batch``` dimension
- $ tanh $ Non Linearity
    - A non linearity for squishing the neuron's activation to the range $[-1, 1]$
    
###### The above layers are repeated until the subsequent pairs are all merged, resulting in a flat linear layer.

- **Output Layer**
    - The output layer is a linear layer layer with ```vocab_size``` number of neurons, representing the next likely token


    
## Hyperparameters

Following is the list of hyperparameters to be tuned:-

- ```block_size```
    - Indicates the context length or how long back to look into the sequence,
- ```n_embd```
    - Indicates the dimension of the embedding vector for each token
- ```n_hidden```
    - The number of hidden units(neurons) in each linear layer
    

## Optimizer Used

The optimizer used for training the model is ```AdamW``` optimizer

```python
torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False)
```

## Loss Metric

The loss metric used is ```CrossEntropy Loss```

```python
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
```

code for the model:
```python
model = nn.Sequential(
            nn.Embedding(vocab_size, n_embd),
            nn.Flatten(), nn.Unflatten(1, (n_embd, block_size)),
    
            nn.Conv1d(n_embd, n_hidden//4, kernel_size=k_size, stride=2, padding="valid"),
            nn.Tanh(),
            nn.BatchNorm1d(n_hidden//4),

            nn.Conv1d(n_hidden//4, n_hidden//2, kernel_size=k_size, stride=2, padding="valid"),
            nn.Tanh(),
            nn.BatchNorm1d(n_hidden//2),

            nn.Conv1d(n_hidden//2, n_hidden, kernel_size=k_size, stride=2, padding="valid"),
            nn.Tanh(),
            nn.BatchNorm1d(n_hidden),
            nn.Flatten(),
            nn.Dropout(dropout),

            nn.Linear(n_hidden, vocab_size)
        )
```
''', unsafe_allow_html  = True)
