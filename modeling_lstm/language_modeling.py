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

        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.to_dict()
        print(config)
        self.vocab_size = self.config['vocab_size']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_size = self.config['hidden_size']
        
        self.num_layers = self.config['num_layers']
        self.act_on = self.config['act_on']
        self.act_type = self.config['act_type']
        self.constant_depth = self.config['constant_depth']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if not self.act_on:
            # self.lstm_layer = nn.ModuleList([nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)])
            self.lstm_layer = nn.ModuleList([nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=1, batch_first=True) for _ in range(self.num_layers)])
        elif self.act_type == "layer":
            self.lstm_layer = nn.ModuleList([nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=1, batch_first=True) for _ in range(self.num_layers)])
        elif self.act_type == 'model':
            self.lstm_layer = nn.ModuleList([nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)])
        else:
            raise NotImplementedError

        # self.lstm_layer = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)

        if self.act_on:
            self.max_hop = self.config['max_hop']
            for i in range(len(self.lstm_layer)):
                self.lstm_layer[i] = AdaptiveLayerWrapper(self.lstm_layer[i], self.hidden_size, self.max_hop, self.constant_depth)


        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    def forward(self, x):
        # print(x[0], x[0].shape)
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
       
  
        total_remainders = []
        total_n_updates = []

        for i in range(len(self.lstm_layer)):
            if self.act_on:
                embedded, (remainders, n_updates) = self.lstm_layer[i](embedded)
                total_remainders.append(remainders)
                total_n_updates.append(n_updates)
                embedded = embedded[0]
            else:
                embedded, _ = self.lstm_layer[i](embedded)

        logits = self.fc(embedded)  # Shape: (batch_size, seq_len, vocab_size)
        out = Munch(logits=logits, n_updates=total_n_updates, remainders=total_remainders)
        # print(out.logits[0], out.logits[0].shape)
        return out
    

 