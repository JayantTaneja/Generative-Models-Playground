import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import streamlit as st
import random
random.seed(42)

class WaveNet:
    def set_parameters(
            self, 
            words = [],chars = [], 
            vocab_size = None, 
            stoi = None, itos = None, 
            block_size:int = 8, 
            n_embd :int = 24, 
            n_hidden:int =  300, 
            dropout :int= 0.2, 
            k_size :int= 2
        )->None:
        '''
        Sets various parameters for the model
        '''

        self.words = words
        self.chars = chars
        self.vocab_size = vocab_size
        self.stoi = stoi
        self.itos = itos
        self.block_size = block_size
        self.n_embd =  n_embd
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.k_size = k_size

    def __init__(
            self, 
            lr : int = 0.001, 
            out_dir:str = "model.pth", 
            pre_trained :bool = False,  
            device : str = 'cpu', 
            **kwargs
        )->None:

        self.device = device
        self.out_dir = out_dir
        self.set_parameters()

        if("hparameters" in kwargs):
            self.set_parameters(**kwargs["hparameters"])

        if(pre_trained):
            checkpoint = torch.load(kwargs["checkpoint"],  map_location=torch.device('cpu'))
            self.set_parameters(**checkpoint["hparameters"])
        
        else:
            self.load_dataset(kwargs["dataset_path"])
        
        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.n_embd),
            nn.Flatten(), nn.Unflatten(1, (self.n_embd, self.block_size)),
    
            nn.Conv1d(self.n_embd, self.n_hidden//4, kernel_size=self.k_size, stride=2, padding="valid"),
            nn.Tanh(),
            nn.BatchNorm1d(self.n_hidden//4),

            nn.Conv1d(self.n_hidden//4, self.n_hidden//2, kernel_size=self.k_size, stride=2, padding="valid"),
            nn.Tanh(),
            nn.BatchNorm1d(self.n_hidden//2),

            nn.Conv1d(self.n_hidden//2, self.n_hidden, kernel_size=self.k_size, stride=2, padding="valid"),
            nn.Tanh(),
            nn.BatchNorm1d(self.n_hidden),
            nn.Flatten(),
            nn.Dropout(self.dropout),

            nn.Linear(self.n_hidden, self.vocab_size)
        )
        parameters = self.model.parameters()
        self.parameter_count = sum(p.nelement() for p in parameters)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.model.to(self.device)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

        if pre_trained:
            self.load_checkpoint(kwargs["checkpoint"])

    def load_dataset(self, path:str):
        '''
        Loads the words list with the data from path
        '''

        self.words = open(path, 'r').read().splitlines()
        self.chars = sorted(list(set(''.join(self.words))))

        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0 #used as either stop of sequence or for padding
        
        self.itos = {i:s for s,i in self.stoi.items()}
        
        self.vocab_size = len(self.itos)

        random.shuffle(self.words)

    def _build_dataset_utility(self, words_list):
        
        X, Y = [], []

        for w in words_list:
            context = [0] * self.block_size

            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append

        X = torch.tensor(X, device = self.device)
        Y = torch.tensor(Y, device = self.device)
        return X, Y
    
    def build_dataset(self, path = None):
        if path:
            self.load_dataset(path)

        #build train and test splits of data
        n1 = int(0.8*len(self.words))
        n2 = int(0.9*len(self.words))
        
        self.Xtr, self.Ytr = self._build_dataset_utility(self.words[:n1])
        self.Xdev,self.Ydev = self._build_dataset_utility(self.words[n1:n2])
        self.Xte, self.Yte = self._build_dataset_utility(self.words[n2:])


    def train(
            self, 
            max_steps : int = 200000, 
            batch_size : int = 32, 
            track_every : int = 5000,
            save_every : int = 11000, 
            chart : st.line_chart = None
        )->None:
        
        self.build_dataset()
        self.model.train()
        
        curr_best_loss = float("inf")
        stbar = st.progress(0)
        stbar_text = st.text("0 iterations complete")

        for i in range(max_steps):        
            
            # minibatch construct
            ix = torch.randint(0, self.Xtr.shape[0], (batch_size,))
            Xb, Yb = self.Xtr[ix], self.Ytr[ix] # batch X,Y

            # forward pass
            logits = self.model(Xb)
            loss = self.loss_fn(logits, Yb) # loss function
            
            # backward pass
            for param in self.model.parameters():
                param.grad = None
            loss.backward()
            self.optimizer.step()
            
            # track stats
            if i % track_every == 0 or i==max_steps-1:
                loss_v = self.split_loss('val')
                loss_t = self.split_loss('train')
                
                chart_data = pd.DataFrame({
                    "Train Loss" : [loss_t],
                    "Valid Loss" : [loss_v]
                }, index = [i])
                chart.add_rows(chart_data)
            
            if (loss_v < curr_best_loss) and (i%save_every == 0 or i == max_steps - 1):
                curr_best_loss = loss_v
                stbar_text.text(f"{i}/{max_steps} complete, saving checkpoint...")
                self.save_checkpoint(self.out_dir)
                stbar_text.text(f"{i}/{max_steps} complete, checkpoint saved")
                time.sleep(1)
        
            stbar.progress((i)/max_steps)
            stbar_text.text(f"{i+1}/{max_steps} iterations complete, \ntrain loss : {loss_t:.4f}, \nval   loss : {loss_v:.4f} ")
       
    @torch.no_grad()
    def split_loss(self, split):
        self.model.eval()
        x,y = {
            'train':(self.Xtr, self.Ytr),
            'val':  (self.Xdev, self.Ydev),
            'test': (self.Xte, self.Yte),
        }[split]

        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.model.train()
        
        return loss.item()

    def vis_sample(self, start = ""):
        self.model.eval()

        out = []
        context = [0] * self.block_size # initialize with all ...
        for x in start:
            out.append(self.stoi[x])
            
            if len(start)<=8:
                context_string = start
            else:
                context_string = start[-8::]
                
            for idx in range(1, len(context_string)+1):
                context[-idx] = self.stoi[context_string[-idx]]
                
                
        probabilities = []
        while True:

            # forward pass the neural net
            logits = self.model(torch.tensor([context], device = self.device))
            probs = F.softmax(logits, dim=1)
            
            probabilities.append(probs)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()

            # shift the context window and track the samples
            context = context[1:] + [ix]

            out.append(ix)

            # if we sample the special '.' token, break
            if ix == 0:
                break

        return probabilities, ''.join(self.itos[i] for i in out)[:-1]
    
    def sample(self, num_samples = 20, start = ""):
        self.model.eval()

        results = []
        for _ in range(num_samples):
            out = []
            context = [0] * self.block_size # initialize with all ...
            
            for x in start:
                out.append(self.stoi[x])
            
            if len(start)<=8:
                context_string = start
            else:
                context_string = start[-8::]
            for idx in range(1, len(context_string)+1):
                context[-idx] = self.stoi[context_string[-idx]]
                
            while True:

                # forward pass the neural net
                logits = self.model(torch.tensor([context], device = self.device))
                probs = F.softmax(logits, dim=1)
            
                # sample from the distribution
                ix = torch.multinomial(probs, num_samples=1).item()
                
                # shift the context window and track the samples
                context = context[1:] + [ix]
                
                out.append(ix)
                
                # if we sample the special '.' token, break
                if ix == 0:
                    break
            
            results.append(''.join(self.itos[i] for i in out)[:-1])
        return results

    def save_checkpoint(self, path:str):
        torch.save({
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : self.optimizer.state_dict(),
                'hparameters' : {
                    'block_size':self.block_size, 
                    'n_embd':self.n_embd, 
                    'n_hidden':self.n_hidden, 
                    'stoi':self.stoi, 
                    'itos':self.itos, 
                    'chars':self.chars, 
                    'vocab_size':self.vocab_size
                }
            }, path)

    def load_checkpoint(self, path:str):
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()

