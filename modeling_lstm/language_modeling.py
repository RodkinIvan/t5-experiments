import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from munch import Munch

from modeling_lstm.act_utils import AdaptiveLayerWrapper


# PyTorch Lightning Module to train the Double LSTM
class DoubleLSTMModel(nn.Module):
    def __init__(self, config):
        super(DoubleLSTMModel, self).__init__()

        self.config = config.to_dict()
        print(config)
        self.vocab_size = self.config['vocab_size']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_size = self.config['hidden_size']
        
        self.num_layers = self.config['num_layers']
        self.act_on = self.config['act_on']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.ModuleList([nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=1, batch_first=True) for _ in range(self.num_layers)])

        if self.act_on:
            self.max_hop = self.config['max_hop']
            for i in range(len(self.lstm)):
                self.lstm[i] = AdaptiveLayerWrapper(self.lstm[i], self.hidden_size, self.max_hop)
            
        self.activation = nn.ReLU()

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss() 

    def forward(self, x):
        # print(x[0], x[0].shape)
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
       
  
        self.remainders = []
        self.n_updates = []
        print(self.num_layers)
        for i in range(self.num_layers):

            embedded, (remainders, n_updates) = self.lstm[i](embedded)
            print(embedded)
            embedded = self.activation(embedded[0])
            self.remainders.append(remainders)
            self.n_updates.append(n_updates)
        
        print(len(self.remainders))
        # self.remainders = torch.mean(torch.stack(self.remainders, dim=1), dim=1)
        # self.n_updates = torch.mean(torch.stack(self.n_updates, dim=1), dim=1)

        # self.n_updates = self.n_updates.detach().cpu()
        # self.remainders = self.remainders.detach().cpu()


        logits = self.fc(embedded)  # Shape: (batch_size, seq_len, vocab_size)
        out = Munch(logits=logits, n_updates=self.n_updates, remainders=self.remainders)
        # print(out.logits[0], out.logits[0].shape)
        return out
    

 