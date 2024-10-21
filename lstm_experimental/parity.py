#!/usr/bin/env python
"""
Implements the Parity task from Graves 2016: determining the parity of a
statically-presented binary vector.
"""

import argparse
import random
from typing import Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import models


# Dataset Definition
class ParityDataset(torch.utils.data.IterableDataset):
    """
    An infinite IterableDataset for binary parity problems.
    """
    def __init__(self, bits: int):
        if bits <= 0:
            raise ValueError("bits must be at least one.")
        self.bits = bits

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> Tuple[torch.Tensor, torch.Tensor]:
        vec = torch.zeros(self.bits, dtype=torch.float32)
        num_bits = random.randint(1, self.bits)
        bits = torch.randint(2, size=(num_bits,)) * 2 - 1
        vec[:num_bits] = bits
        parity = (bits == 1).sum() % 2
        return vec, parity.type(torch.float)


# Data Module Definition
class ParityDataModule(pl.LightningDataModule):
    def __init__(self, bits: int, batch_size: int, num_workers: int):
        super().__init__()
        self.bits = bits
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = ParityDataset(bits=self.bits)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


# Model Definition
class ParityModel(pl.LightningModule):
    def __init__(self, bits: int, hidden_size: int, time_penalty: float,
                 learning_rate: float, time_limit: int):
        super().__init__()
        self.save_hyperparameters()
        self.cell = models.AdaptiveRNNCell(
            input_size=bits,
            hidden_size=hidden_size,
            time_penalty=time_penalty,
            time_limit=time_limit
        )
        self.output_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, binary_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, ponder_cost, steps = self.cell(binary_vector)
        logits = self.output_layer(hidden)
        return logits.squeeze(1), ponder_cost, steps

    def training_step(self, batch, _):
        vectors, targets = batch
        logits, ponder_cost, steps = self.forward(vectors)

        log_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss = log_loss + ponder_cost

        # Logging values directly to Lightning
        self.log("loss/total", loss)
        self.log("loss/classification", log_loss)
        self.log("loss/ponder", ponder_cost)
        self.log("accuracy", (logits > 0).eq(targets).float().mean())
        self.log("steps", steps.float().mean())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Main function for setting up and running the training process
def main(args):
    # Initialize Data Module
    data_module = ParityDataModule(
        bits=args.bits,
        batch_size=args.batch_size,
        num_workers=args.data_workers
    )

    # Initialize Model
    model = ParityModel(
        bits=args.bits,
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        learning_rate=args.learning_rate,
        time_limit=args.time_limit
    )

    # Setup WandB Logger
    wandb_logger = WandbLogger(project="parity_task", name="parity_experiment")

    # Setup Trainer with correct accelerator and devices
    if args.devices > 0:
        accelerator = "gpu"
        devices = args.devices
    else:
        accelerator = "cpu"
        devices = 1  # Use 1 device for CPU instead of 0

    trainer = Trainer(
        max_steps=args.max_steps,
        max_epochs=-1,  # To use max_steps as stopping criteria
        logger=wandb_logger,
        log_every_n_steps=100,
        accelerator=accelerator,
        devices=devices
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


# Argument Parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Parity Task Training")
    parser.add_argument("--max-steps", type=int, default=200000, help="Maximum training steps")
    parser.add_argument("--bits", type=int, default=16, help="Number of bits in the binary vector")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the RNN cell")
    parser.add_argument("--time_penalty", type=float, default=1e-3, help="Penalty for time steps in ACT")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--time_limit", type=int, default=20, help="Maximum time steps for the RNN cell")
    parser.add_argument("--data_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--devices", type=int, default=0, help="Number of devices (GPUs/CPUs) to use for training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
