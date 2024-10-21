#!/usr/bin/env python
"""
Implements the Addition task from Graves 2016: adding a collection of decimal
numbers. See Section 3.3 in the paper for full details.
"""

import argparse
import math
import random
import string
from typing import List, Tuple

import pytorch_lightning as pl
import torch

from pytorch_adaptive_computation_time import models

# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class AdditionDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    An infinite IterableDataset for addition problems.

    Examples are pairs (sequence of numbers, sequence of sums), where:
     - Each number is a vector of concatenated one-hot encoded decimal digits
     - Each target is the sum of all prior numbers in the sequence, as one
       11-way classification for each decimal digit (or empty space) in the
       output.

    Parameters
    ----------
    sequence_length
        The length of sequence to add.
    max_digits
        The maximum number of decimal digits each number can have.
    """

    sequence_length: int
    max_digits: int
    feature_size: int
    target_size: int

    NUM_DIGITS = 10
    NUM_CLASSES = 11
    EMPTY_CLASS = 10  # Class index of "empty" digits (beyond the number's end)
    MASK_VALUE = -100

    EMPTY_TOKEN = "-"
    VOCABULARY = string.digits + EMPTY_TOKEN

    def __init__(self, sequence_length: int, max_digits: int):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be at least 1.")
        if max_digits <= 0:
            raise ValueError("max_digits must be at least 1.")

        self.sequence_length = sequence_length
        self.max_digits = max_digits
        self.feature_size = self.NUM_DIGITS * max_digits

        # Find the number of digits we need to allocate to the target.
        # Sum is bounded by sequence_length * (10^max_digits); requires
        # log10(sequence_length * (10^max_digits))
        # = max_digits + log10(sequence_length) digits to represent.
        self.target_size = max_digits + math.ceil(math.log10(sequence_length))

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self):
        cumsum = 0
        features = torch.empty(
            [self.sequence_length, self.feature_size], dtype=torch.float32
        )
        targets = torch.empty(
            [self.sequence_length, self.target_size], dtype=torch.long
        )

        for idx in range(self.sequence_length):
            number, digits = self._get_number_and_digits()
            cumsum += number
            digits_onehot = list(map(self._onehot, digits))
            features[idx] = torch.cat(digits_onehot)

            if idx > 0:
                sum_digits = [string.digits.index(digit) for digit in str(cumsum)]
                missing_digits = self.target_size - len(sum_digits)
                sum_digits.extend([self.EMPTY_CLASS] * missing_digits)
                targets[idx] = torch.as_tensor(sum_digits, dtype=torch.long)
            else:  # Mask out first targets
                targets[idx, :] = self.MASK_VALUE

        return features, targets

    def _get_number_and_digits(self) -> Tuple[int, List[str]]:
        num_digits = random.randint(1, self.max_digits)
        number = 0
        digits = []
        for _ in range(num_digits):
            digit = random.randint(0, 9)
            number = number * 10 + digit
            digits.append(str(digit))

        digits.extend(["-"] * (self.max_digits - num_digits))
        return number, digits

    @staticmethod
    def _onehot(token: str) -> torch.Tensor:
        if len(token) != 1 or token not in AdditionDataset.VOCABULARY:
            raise ValueError(f"token must be one of [{AdditionDataset.VOCABULARY}]")

        onehot = torch.zeros(AdditionDataset.NUM_DIGITS, dtype=torch.float32)
        if token != AdditionDataset.EMPTY_TOKEN:
            onehot[int(token)] = 1.0
        return onehot

    @staticmethod
    def _collate_examples(
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, targets = zip(*batch)
        features_padded = torch.nn.utils.rnn.pad_sequence(
            features, padding_value=AdditionDataset.MASK_VALUE
        )
        targets_padded = torch.nn.utils.rnn.pad_sequence(
            targets, padding_value=AdditionDataset.MASK_VALUE
        )
        return features_padded, targets_padded


class AdditionModel(pl.LightningModule):
    """
    An ACT RNN for addition tasks.
    """

    def __init__(
        self,
        sequence_length: int,
        max_digits: int,
        hidden_size: int,
        time_penalty: float,
        batch_size: int,
        learning_rate: float,
        time_limit: int,
        data_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = AdditionDataset(sequence_length, max_digits)
        num_logits = AdditionDataset.NUM_CLASSES * self.dataset.target_size

        self.rnn = models.AdaptiveGRU(
            input_size=self.dataset.feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            time_penalty=time_penalty,
            time_limit=time_limit,
        )
        self.output_layer = torch.nn.Linear(hidden_size, num_logits)

    def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--max-steps", type=int, default=200000)
        parser.add_argument("--sequence-length", type=int, default=5)
        parser.add_argument("--max-digits", type=int, default=5)
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--time_penalty", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--time-limit", type=int, default=20)
        parser.add_argument("--data-workers", type=int, default=1)
        return parser

    def forward(
        self, number_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, _, ponder_cost = self.rnn(number_sequence)
        logits = self.output_layer(hidden)  # Shape: [sequence, batch, num_logits]
        logits = logits.view(
            [
                logits.size(0),
                logits.size(1),
                self.dataset.target_size,
                AdditionDataset.NUM_CLASSES,
            ]
        )
        return logits, ponder_cost

    def training_step(self, batch, _):
        numbers, sums = batch
        logits, ponder_cost = self(numbers)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        log_loss = torch.nn.functional.nll_loss(
            log_probs.view(-1, AdditionDataset.NUM_CLASSES),
            sums.view(-1),
            ignore_index=AdditionDataset.MASK_VALUE,
        )
        loss = log_loss + ponder_cost

        matches = (logits.detach().argmax(-1) == sums)[1:]
        place_accuracy = matches.type(torch.float).mean().cpu()
        sequence_accuracy = matches.all(0).all(-1).type(torch.float).mean().cpu()

        return {
            "loss": loss,
            "log": {
                "loss/total": loss,
                "loss/classification": log_loss,
                "loss/ponder": ponder_cost,
                "accuracy/place": place_accuracy,
                "accuracy/sequence": sequence_accuracy,
            },
        }

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=AdditionDataset._collate_examples,
        )


def get_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return AdditionModel.add_argparse_args(parser)


def main():
    parser = get_base_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = AdditionModel(
        sequence_length=args.sequence_length,
        max_digits=args.max_digits,
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        time_limit=args.time_limit,
        data_workers=args.data_workers,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, row_log_interval=100, max_steps=args.max_steps
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
