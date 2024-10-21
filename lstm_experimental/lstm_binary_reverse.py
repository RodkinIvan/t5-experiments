import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from LM_DoubleLSTM import DoubleLSTMModel

from processing import DataCollatorWithUniformRandomOffsetsForCausalLM_reverse

# Define the token mapping for the Binary Reverse Task
token_mapping = {
    '0': 0,
    '1': 1,
    '=': 2,       # Separator between input and target
    '<PAD>': 3    # Padding token
}
vocab_size = len(token_mapping)

import torch
from torch.utils.data import Dataset

class BinaryReverseDataset(Dataset):
    def __init__(self, data, token_mapping):
        """
        Args:
            data (pd.DataFrame): DataFrame containing columns 'X' and 'Y',
                                 where 'Y' is the reverse of 'X'.
            token_mapping (dict): Mapping from tokens to integers.
        """
        # Ensure that all binary numbers have a length of 40
        self.data = data[data['X'].apply(len) == 40].reset_index(drop=True)
        self.token_mapping = token_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        X = [self.token_mapping[bit] for bit in row['X']]
        Y = [self.token_mapping[bit] for bit in row['Y']]

        # Input sequence: X + '='
        input_ids = X + [self.token_mapping['=']] + Y

        # Labels: Y
        labels = Y

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# from torch.nn.utils.rnn import pad_sequence

# def binary_reverse_collate_fn(batch):
#     """
#     Custom collate function for Binary Reverse Task.
#     Returns:
#         - input_ids: Padded input sequences
#         - attention_mask: Masks indicating non-padded tokens
#         - labels: Padded target sequences
#     """
#     input_ids = [item['input_ids'] for item in batch]
#     labels = [item['labels'] for item in batch]

#     # Pad input_ids
#     padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=token_mapping['<PAD>'])

#     # Create attention masks
#     attention_mask = (padded_input_ids != token_mapping['<PAD>']).long()

#     # Pad labels
#     padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is the default ignore_index in CrossEntropyLoss

#     return {
#         'input_ids': padded_input_ids,
#         'attention_mask': attention_mask,
#         'labels': padded_labels
    # }

from transformers import PreTrainedTokenizer

class BinaryReverseTokenizer(PreTrainedTokenizer):
    def __init__(self, token_mapping):
        # Initialize the PreTrainedTokenizer with necessary variables
        super().__init__(pad_token='<PAD>', eos_token='=', unk_token='<UNK>')
        self.token_mapping = token_mapping
        self.vocab_size = len(token_mapping)
        self.id_to_token = {v: k for k, v in token_mapping.items()}  # Reverse mapping

    def _tokenize(self, text):
        # Tokenize the input string by simply mapping each character to its corresponding token ID
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        # Convert each token to its corresponding ID using the mapping
        return [self.token_mapping.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        # Convert IDs back to tokens
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def build_inputs_with_special_tokens(self, token_ids):
        # For the binary reverse task, we concatenate input_ids with the '=' separator token
        return token_ids + [self.token_mapping['=']]

    def get_vocab(self):
        return self.token_mapping


tokenizer = BinaryReverseTokenizer(token_mapping)

data_collator = DataCollatorWithUniformRandomOffsetsForCausalLM_reverse(tokenizer, mlm=False, max_offset=30)


# DataLoader to load the binary reverse dataset
class BinaryReverseDataModule(pl.LightningDataModule):
    def __init__(self, file_path, batch_size=32, train_split=0.8, val_split=0.1, test_split=0.1):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        # Load data
        data = pd.read_csv(self.file_path, sep='\t')
        full_dataset = BinaryReverseDataset(data)

        # Split dataset
        train_size = int(self.train_split * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=data_collator
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=data_collator
        )

# Train the model for the Binary Reverse Task
def train_model_reverse(file_path):
    embedding_dim = 8
    hidden_size = 8
    batch_size = 32
    learning_rate = 1e-4
    max_epochs = 100
    num_layers = 1

    # Initialize the data module
    data_module = BinaryReverseDataModule(file_path, batch_size=batch_size)

    # Initialize the model
    model = DoubleLSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
        lr=learning_rate
    )

    # Setup WandB Logger
    wandb_logger = WandbLogger(project="bin_reverse_task", name="lstm_reverse_experiment")
    # from lightning.pytorch.callbacks import LearningRateMonitor
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger) #, callbacks=[lr_monitor])

    # Train the model
    trainer.fit(model, data_module)

    # Optionally, you can add testing or saving the model here
    # trainer.test(model, datamodule=data_module)

# Assuming the binary reverse dataset is stored at 'binary_pairs_reverse.tsv'
if __name__ == "__main__":
    train_model_reverse('data/binary_pairs_reverse.tsv')
