import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import WandbLogger


# PyTorch Lightning Module to train the Double LSTM
class DoubleLSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, token_mapping, num_layers=1, lr=0.001):
        super(DoubleLSTMModel, self).__init__()
        self.save_hyperparameters()

        self.token_mapping = token_mapping

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.token_mapping['<PAD>'])  # Ignore padding index

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
       
        output1, _ = self.lstm1(embedded)

        activated_output = self.activation(output1)
    
        output2, _ = self.lstm2(activated_output)

        logits = self.fc(output2)  # Shape: (batch_size, seq_len, vocab_size)
        return logits
    

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['labels']
        label_length = y.shape[1]
        input_length = x.shape[1]

        logits = self(x)

         # Calculate where the Y sequence starts
        batch_size = logits.size(0)
        # logits_y = []
        # y_ground = []
        
        # for i in range(batch_size):
        #     # Start of Y is after X1 + '+' + X2 + '=' which is given by (input_length - len_y)
        #     y_start_idx = input_lengths[i] - y_lengths[i]
        #     y_end_idx = input_lengths[i]  # End at the full sequence length

        #     # Extract logits for the Y sequence
        #     logits_y.append(logits[i, y_start_idx-1:y_end_idx-1, :])
        #     y_ground.append(y[i, :y_lengths[i]])

        logits_y = logits[:, -label_length-1:, :-1]

        logits_y_flat = torch.flatten(logits_y, end_dim=-2)  
        # Flatten the ground truth Y
        y_flat = torch.flatten(y)  
        
      

        # Calculate loss only on valid tokens (non-padding)
        loss = self.loss_fn(logits_y_flat, y_flat)

        # Calculate token-level accuracy
        preds = logits_y_flat.argmax(dim=-1) 
        correct_tokens = (preds == y_flat)
        token_acc = correct_tokens.float().mean()

        # Calculate sequence-level accuracy
        seq_correct = []
        for i in range(batch_size):
            preds_i = logits_y[i].argmax(dim=-1)  # Predicted Y sequence for sample i
            y_i = y[i]  # Ground truth Y sequence for sample i
            is_correct = torch.equal(preds_i, y_i)  # Check if sequences are identical
            seq_correct.append(is_correct)

        seq_acc = torch.tensor(seq_correct, dtype=torch.float).mean()

        # Log the gradient norms
        total_grad_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.log('grad_norm', total_grad_norm)

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_token_acc', token_acc)
        self.log('train_seq_acc', seq_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        # x, y, input_lengths, y_lengths = batch.
        x = batch['input_ids']
        y = batch['labels']
        label_length = y.shape[1]
        input_length = x.shape[1]

        logits = self(x)

        # Calculate where the Y sequence starts
        batch_size = logits.size(0)
        # logits_y = []
        # y_ground = []
        

        logits_y = logits[:, -label_length:-1, :-1]

        # for i in range(batch_size):
        #     # Start of Y is aftor i in range(batch_size):
        #     # Start of Y is after X1 + '+' + X2 + '=' which is given by (input_length - len_y)
        #     y_start_idx = input_lengths[i] - y_lengths[i]
        #     y_end_idx = input_lengths[i]  # End at the full sequence length

        #     # Extract logits for the Y sequence
        #     logits_y.append(logits[i, y_start_idx-1:y_end_idx-1, :])
        #     y_ground.append(y[i, :y_lengths[i]])er X1 + '+' + X2 + '=' which is given by (input_length - len_y)
        #     y_start_idx = input_lengths[i] - y_lengths[i]
        #     y_end_idx = input_lengths[i]  # End at the full sequence length

        #     # Extract logits for the Y sequence
        #     logits_y.append(logits[i, y_start_idx-1:y_end_idx-1, :])
        #     y_ground.append(y[i, :y_lengths[i]])

        # Concatenate the logits for Y across the batch
        
        logits_y_flat = torch.flatten(logits_y, end_dim=-2)  
        # Flatten the ground truth Y
        y_flat = torch.flatten(y)  
        
      

        # Calculate loss only on valid tokens (non-padding)
        loss = self.loss_fn(logits_y_flat, y_flat)

        # Calculate token-level accuracy
        preds = logits_y_flat.argmax(dim=-1)  
        correct_tokens = (preds == y_flat)
        token_acc = correct_tokens.float().mean()

        # Calculate sequence-level accuracy
        seq_correct = []
        for i in range(batch_size):
            preds_i = logits_y[i].argmax(dim=-1)  # Predicted Y sequence for sample i
            y_i = y[i]  # Ground truth Y sequence for sample i
            is_correct = torch.equal(preds_i, y_i)  # Check if sequences are identical
            seq_correct.append(is_correct)

        seq_acc = torch.tensor(seq_correct, dtype=torch.float).mean()
        

        # Log metrics
        self.log('val_loss', loss)
        self.log('val_token_acc', token_acc)
        self.log('val_seq_acc', seq_acc)
        self.log('lr', self.optimizers().param_groups[0]['lr'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

