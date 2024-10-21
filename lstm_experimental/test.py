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
        self.data = data
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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=token_mapping['<PAD>'])  # Ignore padding index

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
    
    def extract_logits_for_y(self, logits, y, input_lengths):
        """
        Extract only the logits corresponding to the target sequence Y by ignoring padding.
        """
        # Calculate length of target sequence Y for each example in the batch
        len_y = torch.tensor([len(target) for target in y])

        # Initialize a list to hold the logits corresponding to Y
        logits_y = []

        # For each sequence in the batch, extract the logits for Y, ignoring padding
        for i, length in enumerate(input_lengths):
            # Compute the start index for Y: it is len(X1) + 1 (for '+') + len(X2) + 1 (for '=')
            y_start_idx = length - len_y[i]
            y_end_idx = length
            logits_y.append(logits[i, y_start_idx:y_end_idx, :])

        # Concatenate logits for Y across the batch
        logits_y = torch.cat(logits_y, dim=0)

        return logits_y

    def training_step(self, batch, batch_idx):
        x, y, input_lengths, y_lengths = batch
        logits = self(x, input_lengths)  # Shape: (batch_size, seq_len, vocab_size)

         # Calculate where the Y sequence starts
        batch_size = logits.size(0)
        logits_y = []
        y_ground = []
        
        for i in range(batch_size):
            # Start of Y is after X1 + '+' + X2 + '=' which is given by (input_length - len_y)
            y_start_idx = input_lengths[i] - y_lengths[i]
            y_end_idx = input_lengths[i]  # End at the full sequence length

            # Extract logits for the Y sequence
            logits_y.append(logits[i, y_start_idx-1:y_end_idx-1, :])
            y_ground.append(y[i, :y_lengths[i]])

        # Concatenate the logits for Y across the batch      
        logits_y_flat = torch.cat(logits_y, dim=0)  
        # Flatten the ground truth Y
        y_flat = torch.cat(y_ground, dim=0)  # Shape: (batch_size * seq_len,)
        
      

        # Calculate loss only on valid tokens (non-padding)
        loss = self.loss_fn(logits_y_flat, y_flat)

        # Calculate token-level accuracy
        preds = logits_y_flat.argmax(dim=-1)  # Shape: (batch_size *seq_len)
        correct_tokens = (preds == y_flat)
        token_acc = correct_tokens.float().mean()

        # Calculate sequence-level accuracy
        seq_correct = []
        for i in range(batch_size):
            preds_i = logits_y[i].argmax(dim=-1)  # Predicted Y sequence for sample i
            y_i = y_ground[i]  # Ground truth Y sequence for sample i
            is_correct = torch.equal(preds_i, y_i)  # Check if sequences are identical
            seq_correct.append(is_correct)

        seq_acc = torch.tensor(seq_correct, dtype=torch.float).mean()

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_token_acc', token_acc)
        self.log('train_seq_acc', seq_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, input_lengths, y_lengths = batch
        logits = self(x, input_lengths)

        # Calculate where the Y sequence starts
        batch_size = logits.size(0)
        logits_y = []
        y_ground = []
        
        for i in range(batch_size):
            # Start of Y is after X1 + '+' + X2 + '=' which is given by (input_length - len_y)
            y_start_idx = input_lengths[i] - y_lengths[i]
            y_end_idx = input_lengths[i]  # End at the full sequence length

            # Extract logits for the Y sequence
            logits_y.append(logits[i, y_start_idx-1:y_end_idx-1, :])
            y_ground.append(y[i, :y_lengths[i]])

        # Concatenate the logits for Y across the batch
        
        logits_y_flat = torch.cat(logits_y, dim=0)  
        # Flatten the ground truth Y
        y_flat = torch.cat(y_ground, dim=0)  # Shape: (batch_size * seq_len,)
        
      

        # Calculate loss only on valid tokens (non-padding)
        loss = self.loss_fn(logits_y_flat, y_flat)

        # Calculate token-level accuracy
        preds = logits_y_flat.argmax(dim=-1)  # Shape: (batch_size *seq_len)
        correct_tokens = (preds == y_flat)
        token_acc = correct_tokens.float().mean()

        # Calculate sequence-level accuracy
        seq_correct = []
        for i in range(batch_size):
            preds_i = logits_y[i].argmax(dim=-1)  # Predicted Y sequence for sample i
            y_i = y_ground[i]  # Ground truth Y sequence for sample i
            is_correct = torch.equal(preds_i, y_i)  # Check if sequences are identical
            seq_correct.append(is_correct)

        seq_acc = torch.tensor(seq_correct, dtype=torch.float).mean()
        
        print('\n\n\n')
        print(correct_tokens)
        print(token_acc)
        print('\n\n\n')

        print(seq_acc)

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

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)

    # Train the model
    trainer.fit(model, data_module)


# Assuming the binary sum dataset is stored at 'binary_pairs_sum.tsv'
train_model('/home/cosmos/VScode Projects/coglab/NLP/pytorch-adaptive-computation-time/data/binary_pairs_sum.tsv')
