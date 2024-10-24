import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from LM_DoubleLSTM import DoubleLSTMModel

from processing.copy_collator import DataCollatorWithUniformRandomOffsetsForCausalLM_copy

from datasets import load_dataset

# Define the token mapping for the Binary Copy Task
token_mapping = {
    '0': 0,
    '1': 1,
    '=': 2,       # Separator between input and target
    '<PAD>': 3    # Padding token
}
vocab_size = len(token_mapping)

import torch
from torch.utils.data import Dataset

class BinaryCopyDataset(Dataset):
    def __init__(self, data, token_mapping, length=40):
        """
        Args:
            data (pd.DataFrame): DataFrame containing columns 'X' and 'Y',
                                 where 'Y' is the Copy of 'X'.
            token_mapping (dict): Mapping from tokens to integers.
        """
        # Ensure that all binary numbers have a length of 40
        # print(data)
        self.length = length
        self.data = data.filter(self.filter_length)
        
        # data[data['X'].apply(len) == length].reset_index(drop=True)
        self.token_mapping = token_mapping


    def filter_length(self, input):
        return len(input["X"]) == self.length
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]

        X = [self.token_mapping[bit] for bit in row['X']]
        Y = [self.token_mapping[bit] for bit in row['Y']]


        # Input sequence: X + '='
        input_ids = X #+ [self.token_mapping['=']] + Y

        # Labels: Y
        labels = Y

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }




from transformers import PreTrainedTokenizer

class BinaryCopyTokenizer(PreTrainedTokenizer):
    def __init__(self, token_mapping):
        # Set the token mapping before calling super().__init__
        self.token_mapping = token_mapping
        
        # Create Copy mapping (id -> token)
        self.id_to_token = {v: k for k, v in token_mapping.items()}
        
        # Now call the parent class initializer with special tokens

        super().__init__(pad_token='<PAD>', eos_token='=', unk_token='<UNK>')

    @property
    def vocab_size(self):
        # Return the length of the token mapping (vocabulary size)
        return len(self.token_mapping)

    def _tokenize(self, text):
        # Tokenize the text (for binary, it’s just splitting into characters)
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        # Convert tokens to IDs using the token_mapping
        return [self.token_mapping.get(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        # Convert token IDs back to tokens
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def build_inputs_with_special_tokens(self, token_ids):
        # Add the separator '=' token at the end of the sequence
        return token_ids + [self.token_mapping['=']]

    def get_vocab(self):
        # Return the token mapping (for Hugging Face internals)
        return self.token_mapping




tokenizer = BinaryCopyTokenizer(token_mapping)

data_collator = DataCollatorWithUniformRandomOffsetsForCausalLM_copy(tokenizer, mlm=False, max_offset=35)


# DataLoader to load the binary Copy dataset
class BinaryCopyDataModule(pl.LightningDataModule):
    def __init__(self, path=None, length=40, batch_size=32, train_split=0.8, val_split=0.1, test_split=0.1):
        super().__init__()
        self.path = path
        self.length = length
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        

    def setup(self, stage=None):

        train = load_dataset(self.path, data_files="binary_copy_train.tsv", delimiter="\t")
        val = load_dataset(self.path, data_files="binary_copy_val.tsv", delimiter="\t")
        test = load_dataset(self.path, data_files="binary_copy_test.tsv", delimiter="\t")

        train.set_format(type='torch')
        val.set_format(type='torch')
        test.set_format(type='torch')

        self.train_dataset = BinaryCopyDataset(train['train'], token_mapping, length=self.length) 
        self.val_dataset = BinaryCopyDataset(val['train'], token_mapping, length=self.length)
        self.test_dataset = BinaryCopyDataset(test['train'], token_mapping, length=self.length)
    
        if self.length == 400:
            # Split dataset
            train_size = int(self.train_split * len(self.test_dataset))
            val_size = int(self.val_split * len(self.test_dataset))
            test_size = len(self.test_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.test_dataset, [train_size, val_size, test_size]
            )




    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=data_collator, num_workers=10
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=data_collator
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=data_collator
        )

# Train the model for the Binary Copy Task
def train_model_Copy(file_path):
    embedding_dim = 8
    hidden_size = 128
    batch_size = 64
    learning_rate = 1e-3
    max_epochs = 100
    num_layers = 1

    SEQ_LENGTH = 400

    

    # data = load_dataset("DaniilOr/copy")
    # print(data.keys())

    # Initialize the data module
    data_module = BinaryCopyDataModule(path="DaniilOr/copy", batch_size=batch_size, length=SEQ_LENGTH)


    # Initialize the model
    model = DoubleLSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        token_mapping=token_mapping,
        num_layers=num_layers,
        hidden_size=hidden_size,
        lr=learning_rate
    )
    model = DoubleLSTMModel.load_from_checkpoint('./checkpoints/binary_copy/lenSEQ_LENGTH=0-best-checkpoint-val_loss=0.00.ckpt')

    # Setup WandB Logger
    wandb_logger = WandbLogger(project="bin_Copy_task", name="lstm_Copy_experiment")
    # from lightning.pytorch.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    

    import os
    from pytorch_lightning.callbacks import ModelCheckpoint

    # Configure the ModelCheckpoint callback to save the best model (with the lowest val_loss)
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/binary_copy/',  # Directory to save the checkpoint
        filename=f'len{SEQ_LENGTH}'+'-best-checkpoint-{val_loss:.2f}',  # Base name for the saved file
        save_top_k=1,  # Save only the top 1 model
        verbose=True,
        monitor='val_loss',  # Monitor validation loss
        mode='min'  # We want the model with the lowest validation loss
    )
    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback]) #, callbacks=[lr_monitor])

    # Train the model
    trainer.fit(model, data_module)

    # Optionally, you can add testing or saving the model here
    # trainer.test(model, datamodule=data_module)

    # trainer.test(model, data_module)

# Assuming the binary Copy dataset is stored at 'binary_pairs_Copy.tsv'
if __name__ == "__main__":
    train_model_Copy('./data/binary_reverse.tsv')
