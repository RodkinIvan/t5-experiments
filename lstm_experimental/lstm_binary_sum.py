import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from LM_DoubleLSTM import DoubleLSTMModel

# Define the token mapping
token_mapping = {
    '0': 0,
    '1': 1,
    '+': 2,
    '=': 3,
    '<PAD>': 4
}
vocab_size = len(token_mapping)

# Dataset class to handle binary sequences
class BinarySumDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): DataFrame containing columns 'X1', 'X2', and 'Y'.
        """
        self.data = data[
            (data['X1'].apply(len) == 40)
        ]
        self.token_mapping = token_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        X1 = [self.token_mapping[bit] for bit in row['X1']]
        X2 = [self.token_mapping[bit] for bit in row['X2']]
        Y = [self.token_mapping[bit] for bit in row['Y']]

        # Input sequence: X1 + '+' + X2 + '='
        input_sequence = X1 + [self.token_mapping['+']] + X2 + [self.token_mapping['=']] + Y

        # Target sequence: Y
        target_sequence = Y

        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)

        return input_tensor, target_tensor

def pad_collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length in each batch.
    """
    inputs, targets = zip(*batch)
    input_lengths = torch.tensor([len(seq) for seq in inputs])

    max_input_length = max(len(seq) for seq in inputs)
    max_target_length = max(len(seq) for seq in targets)
    max_length = max(max_input_length, max_target_length)

    padded_inputs = []
    padded_targets = []
    y_lengths = []

    for inp, tgt in zip(inputs, targets):
        # Pad inputs
        pad_size_inp = max_length - len(inp)
        padded_inp = torch.cat([inp, torch.full((pad_size_inp,), token_mapping['<PAD>'], dtype=torch.long)])
        padded_inputs.append(padded_inp)

        # Pad targets with -100 (ignore_index for CrossEntropyLoss)
        y_lengths.append(len(tgt))
        pad_size_tgt = max_length - len(tgt)
        padded_tgt = torch.cat([tgt, torch.full((pad_size_tgt,), token_mapping['<PAD>'], dtype=torch.long)])
        padded_targets.append(padded_tgt)

    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets, dim=0)
    return padded_inputs, padded_targets, input_lengths, y_lengths

# DataLoader to load the binary sum dataset
class BinarySumDataModule(pl.LightningDataModule):
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
        full_dataset = BinarySumDataset(data)

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
            collate_fn=pad_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=pad_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=pad_collate_fn
        )


# Train the model
def train_model(file_path):
    embedding_dim = 8
    hidden_size = 256
    batch_size = 32
    learning_rate = 1e-4
    max_epochs = 100
    num_layers = 1

    # Initialize the data module
    data_module = BinarySumDataModule(file_path, batch_size=batch_size)

    # Initialize the model
    model = DoubleLSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
        lr=learning_rate
    )

    # Setup WandB Logger
    wandb_logger = WandbLogger(project="bin_sum_task", name="lstm_experiment")
    from lightning.pytorch.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger, callbacks=[lr_monitor])

    # Train the model
    trainer.fit(model, data_module)


# Assuming the binary sum dataset is stored at 'binary_pairs_sum.tsv'
train_model('/home/cosmos/VScode Projects/coglab/NLP/pytorch-adaptive-computation-time/data/binary_pairs_sum.tsv')
