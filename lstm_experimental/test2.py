import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

# Define the token mapping
token_mapping = {
    '0': 0,
    '1': 1,
    '+': 2,
    '=': 3
}
vocab_size = len(token_mapping)

# Dataset class to handle binary sequences
class BinarySumDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): DataFrame containing columns 'X1', 'X2', and 'Y'.
        """
        self.data = data
        self.token_mapping = token_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        X1 = [self.token_mapping[bit] for bit in row['X1']]
        X2 = [self.token_mapping[bit] for bit in row['X2']]
        Y = [self.token_mapping[bit] for bit in row['Y']]

        # Input sequence: X1 + '+' + X2 + '=' + Y
        input_sequence = X1 + [self.token_mapping['+']] + X2 + [self.token_mapping['=']] + Y

        # Target sequence: Y
        target_sequence = Y

        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)

        # Return lengths of X1, X2, and Y for later use
        len_X1 = len(X1)
        len_X2 = len(X2)
        len_Y = len(Y)

        return input_tensor, target_tensor, len_X1, len_X2, len_Y

def pad_collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length in each batch.
    """
    inputs, targets, len_X1s, len_X2s, len_Ys = zip(*batch)
    input_lengths = torch.tensor([len(seq) for seq in inputs])

    max_input_length = max(len(seq) for seq in inputs)
    max_target_length = max(len(seq) for seq in targets)

    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        # Pad inputs
        pad_size_inp = max_input_length - len(inp)
        padded_inp = torch.cat([inp, torch.full((pad_size_inp,), token_mapping['0'], dtype=torch.long)])
        padded_inputs.append(padded_inp)

        # Pad targets with -100 (ignore_index for CrossEntropyLoss)
        pad_size_tgt = max_target_length - len(tgt)
        padded_tgt = torch.cat([tgt, torch.full((pad_size_tgt,), -100, dtype=torch.long)])
        padded_targets.append(padded_tgt)

    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)

    # Also convert lengths to tensors
    len_X1s = torch.tensor(len_X1s)
    len_X2s = torch.tensor(len_X2s)
    len_Ys = torch.tensor(len_Ys)

    return padded_inputs, padded_targets, input_lengths, len_X1s, len_X2s, len_Ys

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

# PyTorch Lightning Module to train the Double LSTM
class DoubleLSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, lr=0.001):
        super(DoubleLSTMModel, self).__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding index

    def forward(self, x, input_lengths):
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output1, _ = self.lstm1(packed_embedded)
        output1, _ = nn.utils.rnn.pad_packed_sequence(packed_output1, batch_first=True)

        activated_output = self.activation(output1)
        packed_activated = nn.utils.rnn.pack_padded_sequence(
            activated_output, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output2, _ = self.lstm2(packed_activated)
        output2, _ = nn.utils.rnn.pad_packed_sequence(packed_output2, batch_first=True)

        logits = self.fc(output2)  # Shape: (batch_size, seq_len, vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, input_lengths, len_X1s, len_X2s, len_Ys = batch
        logits = self(x, input_lengths)  # Shape: (batch_size, seq_len, vocab_size)

        batch_size, seq_len, vocab_size = logits.size()

        # Calculate starting indices of Y in the input sequence
        y_start_idxs = len_X1s + 1 + len_X2s + 1  # '+' and '=' are single tokens

        # Create a mask for positions corresponding to Y in the input sequence
        positions = torch.arange(seq_len, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        y_start_idxs_expanded = y_start_idxs.unsqueeze(1)
        len_Ys_expanded = len_Ys.unsqueeze(1)
        mask = (positions >= y_start_idxs_expanded) & (positions < y_start_idxs_expanded + len_Ys_expanded)

        # Flatten logits and targets, and apply the mask
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = y.view(-1)
        mask_flat = mask.view(-1)

        logits_y_flat = logits_flat[mask_flat]
        targets_y_flat = targets_flat[mask_flat]

        # Calculate loss
        loss = self.loss_fn(logits_y_flat, targets_y_flat)

        # Calculate token-level accuracy
        preds = logits.argmax(dim=-1)
        preds_flat = preds.view(-1)
        preds_y_flat = preds_flat[mask_flat]
        correct_tokens = (preds_y_flat == targets_y_flat)
        token_acc = correct_tokens.float().mean()

        # Calculate sequence-level accuracy
        seq_correct = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
        for i in range(batch_size):
            y_len = len_Ys[i]
            y_start = y_start_idxs[i]
            y_end = y_start + y_len

            if y_len == 0:
                seq_correct[i] = True  # If Y is empty, consider it correct
            else:
                seq_correct[i] = torch.equal(preds[i, y_start:y_end], y[i, :y_len])

        seq_acc = seq_correct.float().mean()

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_token_acc', token_acc)
        self.log('train_seq_acc', seq_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, input_lengths, len_X1s, len_X2s, len_Ys = batch
        logits = self(x, input_lengths)

        batch_size, seq_len, vocab_size = logits.size()

        # Calculate starting indices of Y in the input sequence
        y_start_idxs = len_X1s + 1 + len_X2s + 1  # '+' and '=' are single tokens

        # Create a mask for positions corresponding to Y in the input sequence
        positions = torch.arange(seq_len, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        y_start_idxs_expanded = y_start_idxs.unsqueeze(1)
        len_Ys_expanded = len_Ys.unsqueeze(1)
        mask = (positions >= y_start_idxs_expanded) & (positions < y_start_idxs_expanded + len_Ys_expanded)

        # Flatten logits and targets, and apply the mask
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = y.view(-1)
        mask_flat = mask.view(-1)

        logits_y_flat = logits_flat[mask_flat]
        targets_y_flat = targets_flat[mask_flat]

        # Calculate loss
        loss = self.loss_fn(logits_y_flat, targets_y_flat)

        # Calculate token-level accuracy
        preds = logits.argmax(dim=-1)
        preds_flat = preds.view(-1)
        preds_y_flat = preds_flat[mask_flat]
        correct_tokens = (preds_y_flat == targets_y_flat)
        token_acc = correct_tokens.float().mean()

        # Calculate sequence-level accuracy
        seq_correct = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
        for i in range(batch_size):
            y_len = len_Ys[i]
            y_start = y_start_idxs[i]
            y_end = y_start + y_len

            if y_len == 0:
                seq_correct[i] = True  # If Y is empty, consider it correct
            else:
                seq_correct[i] = torch.equal(preds[i, y_start:y_end], y[i, :y_len])

        seq_acc = seq_correct.float().mean()

        # Log metrics
        self.log('val_loss', loss)
        self.log('val_token_acc', token_acc)
        self.log('val_seq_acc', seq_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Train the model
def train_model(file_path):
    embedding_dim = 8
    hidden_size = 256
    batch_size = 32
    learning_rate = 0.001
    max_epochs = 100
    num_layers = 2

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

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)

    # Train the model
    trainer.fit(model, data_module)

# Assuming the binary sum dataset is stored at 'binary_pairs_sum.tsv'
train_model('/home/cosmos/VScode Projects/coglab/NLP/pytorch-adaptive-computation-time/data/binary_pairs_sum.tsv')
