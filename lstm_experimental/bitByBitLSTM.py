import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import WandbLogger


# Define the LSTM model for binary sum
class BinarySumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinarySumLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for output

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)  # Pass the packed input through LSTM
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(lstm_out)  # Apply the fully connected layer to the outputs
        return output

# Dataset class to handle binary sequences
class BinarySumDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): DataFrame containing columns 'X1', 'X2', and 'Y'.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        X1 = torch.tensor([int(bit) for bit in row['X1']], dtype=torch.float32)
        X2 = torch.tensor([int(bit) for bit in row['X2']], dtype=torch.float32)
        Y = torch.tensor([int(bit) for bit in row['Y']], dtype=torch.float32)

        # Concatenate X1 and X2 as two features per time step
        X = torch.stack((X1, X2), dim=1)  # Shape: (sequence_length, 2)
        
        return X, Y  # Y as target (shape: (sequence_length,))

def pad_collate_fn(batch):
    """
    Custom collate function to pad binary sequences to the same length in each batch.
    """
    X_batch, Y_batch = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in X_batch])
    max_len = lengths.max()

    # Pad sequences to the max length
    X_padded = [F.pad(x, (0, 0, 0, max_len - x.size(0)), "constant", 0) for x in X_batch]
    Y_padded = [F.pad(y, (0, max_len - y.size(0)), "constant", 0) for y in Y_batch]

    X_padded = torch.stack(X_padded)
    Y_padded = torch.stack(Y_padded)

    return X_padded, Y_padded, lengths

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

# PyTorch Lightning Module to train the LSTM
class DoubleLSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(DoubleLSTMModel, self).__init__()
        self.lstm_model = BinarySumLSTM(input_size, hidden_size, output_size)
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # Use reduction='none' for masking

    def forward(self, x, lengths):
        return self.lstm_model(x, lengths)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self(x, lengths)  # Shape: (batch_size, seq_len, 1)
        y_hat = y_hat.squeeze(-1)  # Shape: (batch_size, seq_len)

        # Create mask
        mask = torch.arange(y.size(1), device=y.device)[None, :] < lengths[:, None]

        # Compute loss only on valid positions
        loss = self.loss_fn(y_hat, y.float())
        loss = (loss * mask.float()).sum() / mask.sum()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self(x, lengths).squeeze(-1)

        # Apply a sigmoid to convert the logits to probabilities
        y_pred = torch.sigmoid(y_hat)

        # Create mask
        mask = torch.arange(y.size(1), device=y.device)[None, :] < lengths[:, None]

        # Apply the mask to filter valid positions
        y_pred = y_pred * mask.float()

        # Threshold the predictions to get binary predictions (0 or 1)
        y_pred_binary = (y_pred > 0.5).float()

        # Calculate token-level accuracy (average accuracy over valid tokens)
        token_acc = ((y_pred_binary == y) * mask).float().sum() / mask.sum()

        # Calculate sequence-level accuracy (correct if all tokens in a sequence match)
        seq_acc = ((y_pred_binary == y).all(dim=1)).float().mean()

        # Compute loss only on valid positions
        loss = self.loss_fn(y_hat, y.float())
        loss = (loss * mask.float()).sum() / mask.sum()

        # Log validation loss, token-level accuracy, and sequence-level accuracy
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_token_acc", token_acc, on_epoch=True)
        self.log("val_seq_acc", seq_acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Train the model
def train_model(file_path):
    input_size = 2  # Two features at each time step: X1 and X2
    hidden_size = 128  # Adjusted hidden size for better learning capacity
    output_size = 1  # The output is a binary sum (1 bit at each time step)
    batch_size = 32
    learning_rate = 0.001
    max_epochs = 20

    # Initialize the data module
    data_module = BinarySumDataModule(file_path, batch_size=batch_size)

    # Initialize the model
    model = DoubleLSTMModel(input_size, hidden_size, output_size, lr=learning_rate)

    # Setup WandB Logger
    wandb_logger = WandbLogger(project="bin_sum_task", name="lstm_experiment")

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)

    # Train the model
    trainer.fit(model, data_module)

# Assuming the binary sum dataset is stored at 'binary_pairs_sum.tsv'
train_model('/home/cosmos/VScode Projects/coglab/NLP/pytorch-adaptive-computation-time/data/binary_pairs_sum.tsv')
