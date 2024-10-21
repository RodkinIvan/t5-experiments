"""
Implements adaptive computation time for RNNs from
`Graves 2016 <https://arxiv.org/abs/1603.08983/>`_.

This module offers 3 interfaces:
 - Adaptive*, which mimics the torch.nn.RNN interface.
 - Adaptive*Cell, which mimics the torch.nn.RNNCell interface.
 - AdaptiveCellWrapper, which wraps any RNN cell to add adaptive computation time.

LSTMs and bidirectional networks are not currently implemented.
"""

import sys
from typing import List, Optional, Tuple

import torch


class AdaptiveCellWrapper(torch.nn.Module):
    """
    Wraps an RNN cell to add adaptive computation time.

    Note that the cell will need an input size of 1 plus the desired input size, to
    allow for the extra first-step flag input.

    Parameters
    ----------
    cell
        The cell to wrap.
    time_penalty
        How heavily to penalize the model for thinking too long. Tau in Graves 2016.
    initial_halting_bias
        Value to initialize the halting unit's bias to. Recommended to set this
        to a negative number to prevent long ponder sequences early in training.
    ponder_epsilon
        When the halting values sum to more than 1 - ponder_epsilon, stop computation.
        Used to enable halting on the first step.
    time_limit
        Hard limit for how many substeps any computation can take. Intended to prevent
        overly-long computation early-on. M in Graves 2016.
    """

    time_penalty: float
    ponder_epsilon: float
    time_limit: int

    _cell: torch.nn.modules.RNNCellBase
    _halting_unit: torch.nn.Module

    def __init__(
        self,
        cell: torch.nn.RNNCellBase,
        time_penalty: float,
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
    ):
        super().__init__()

        if time_penalty <= 0:
            raise ValueError("time_penalty must be positive.")
        if ponder_epsilon < 0 or ponder_epsilon >= 1:
            raise ValueError(
                "ponder_epsilon must be between 0 (inclusive) and 1 (exclusive)"
            )

        self.time_penalty = time_penalty
        self.ponder_epsilon = ponder_epsilon
        self.time_limit = time_limit

        self._cell = cell
        self._halting_unit = torch.nn.Sequential(
            torch.nn.Linear(cell.hidden_size, 1),
            torch.nn.Flatten(),  # type: ignore
            torch.nn.Sigmoid(),
        )

        torch.nn.init.constant_(self._halting_unit[0].bias, initial_halting_bias)

    def forward(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one timestep of the RNN, which may correspond to several internal steps.

        Parameters
        ----------
        inputs
            Tensor containing input features.

            *Shape: (batch, input_size)*
        hidden
            Initial hidden value for the wrapped cell. If not provided, relies on
            the wrapped cell to provide its own initial value.

            *Shape: (batch, hidden_size)*

        Returns
        -------
        next_hiddens : torch.Tensor
            The hidden state for this timestep.

            *Shape: (batch, hidden_size)*
        ponder_cost : torch.Tensor
            The ponder cost for this timestep.

            *Shape: () (scalar)*
        ponder_steps : torch.Tensor
            The number of ponder steps each element in the batch took.

            *Shape: (batch)*
        """
        batch_size = inputs.size(0)
        budget = torch.ones((batch_size, 1), device=inputs.device) - self.ponder_epsilon

        # Accumulate intermediate values throughout the ponder sequence
        total_hidden = torch.zeros(
            [batch_size, self._cell.hidden_size], device=inputs.device
        )
        total_remainder = torch.zeros_like(budget)
        total_steps = torch.zeros_like(budget)

        # Extend input with first-step flag
        first_inputs = torch.cat([inputs, torch.ones_like(budget)], dim=1)
        rest_inputs = torch.cat([inputs, torch.zeros_like(budget)], dim=1)

        # Sum of halting values for all steps except the last
        halt_accum = torch.zeros_like(budget)
        continuing_mask = torch.ones_like(budget, dtype=torch.bool)

        for step in range(self.time_limit - 1):
            step_inputs = first_inputs if step == 0 else rest_inputs
            hidden = self._cell(step_inputs, hidden)

            step_halt = self._halting_unit(hidden)
            masked_halt = continuing_mask * step_halt

            with torch.no_grad():
                halt_accum += masked_halt

            # Select indices ending at this step
            ending_mask = continuing_mask.bitwise_and(halt_accum + step_halt > budget)
            continuing_mask = continuing_mask.bitwise_and(ending_mask.bitwise_not())
            total_steps += continuing_mask

            # 3 cases, computed in parallel by masking batch elements:
            # - Continuing computation: weight new values by the halting probability
            # - Ending at this step: weight new values by the remaining budget
            # - Ended previously: no new values (accumulate zero)
            masked_remainder = ending_mask * (1 - halt_accum)
            combined_mask = masked_halt + masked_remainder

            total_hidden = total_hidden + (combined_mask * hidden)
            total_remainder = total_remainder + masked_halt

            # If all batch indices are done, stop iterations early
            if not continuing_mask.any().item():
                break

        else:  # Some elements ran past the hard limit
            # continuing_mask now selects for these elements
            masked_remainder = continuing_mask * (1 - halt_accum)
            total_hidden = total_hidden + (masked_remainder * hidden)

        # Equal gradient to the true cost; maximize all halting values except the last
        ponder_cost = -1 * self.time_penalty * total_remainder.mean()

        return total_hidden, ponder_cost, total_steps + 1


class _CellUnrollWrapper(torch.nn.Module):
    """
    Wraps an adaptive cell into a torch.nn.RNN-style interface.

    NOTE: because the adaptive cell requires dynamic computation, this does not
    support packed sequences and will not be any more efficient than running the
    cell in a loop (e.g. you don't get the full cuDNN benefits).
    This mainly exists for convenience.

    Parameters
    ----------
    layer_cells
        One cell per layer in the network.
    batch_first
        If True, expects the first dimension of each sequence to be the batch axis
        and the second to be the sequence axis.
    dropout
        Amount of dropout to apply to the output of each layer except the last.
    """

    batch_first: bool
    _layer_cells: torch.nn.ModuleList
    _dropout_layer: Optional[torch.nn.Dropout]

    def __init__(
        self,
        layer_cells: List[AdaptiveCellWrapper],
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not layer_cells:
            raise ValueError("layer_cells must be nonempty.")

        self.batch_first = batch_first
        self._layer_cells = torch.nn.ModuleList(layer_cells)

        if dropout:
            self._dropout_layer = torch.nn.Dropout(dropout)
        else:
            self._dropout_layer = None

    def _apply_layers(
        self, inputs: torch.Tensor, layer_hiddens: List[Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Apply the stack of RNN cells (one per layer) to one timestep of inputs
        and hiddens.

        Parameters
        ----------
        inputs
            A tensor of inputs for this timestep.

            *Shape: (batch, dims)*.
        layer_hiddens
            One (optional) hidden state per layer.

        Returns
        -------
        outputs : torch.Tensor
            The output of the layer stack for this timestep.
        next_hiddens : List[torch.Tensor]
            The layer hidden states for the next timestep.
        ponder_cost : torch.Tensor
            The total ponder cost for this timestep.
        """
        total_ponder_cost = torch.tensor(0.0, device=inputs.device)
        next_hiddens = []

        for i, (cell, hidden) in enumerate(zip(self._layer_cells, layer_hiddens)):
            if self._dropout_layer and i > 0:  # Applies dropout between every 2 layers
                inputs = self._dropout_layer(inputs)

            inputs, ponder_cost, _ = cell(inputs, hidden)
            next_hiddens.append(inputs)
            total_ponder_cost = total_ponder_cost + ponder_cost

        return inputs, next_hiddens, total_ponder_cost  # type: ignore

    def forward(
        self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs
            Input to the network.

            *Shape: (seq_len, batch, input_size)*, or *(batch, seq_len, input_size)*
            if batch_first is True.
        hidden
            Initial hidden state for each element in the batch, as a tensor of

            *Shape: (num_layers \* num_directions, batch, hidden_size)*

        Returns
        -------
        output : torch.Tensor
            The output features from the last layer of the RNN for each timestep.

            *Shape: (seq_len, batch, hidden_size)*, or *(batch, seq_len, hidden_size)*
            if batch_first is True.
        hidden : torch.Tensor
            The hidden state for the final step.

            *Shape: (num_layers, batch, hidden_size)*
        ponder_cost : torch.Tensor
            The total ponder cost for this sequence.

            *Shape: ()*
        """
        if self.batch_first:
            inputs = torch.transpose(inputs, 0, 1)

        timesteps, _, _ = inputs.shape

        if hidden is None:
            layer_hiddens = [None for _ in self._layer_cells]
        else:
            layer_hiddens = [h.squeeze(0) for h in hidden.split(1)]

        total_ponder_cost = torch.tensor(0.0, device=inputs.device)
        outputs = []

        for timestep in range(timesteps):
            output, layer_hiddens, ponder_cost = self._apply_layers(  # type: ignore
                inputs[timestep, :, :], layer_hiddens  # type: ignore
            )
            outputs.append(output)
            total_ponder_cost = total_ponder_cost + ponder_cost

        all_outputs = torch.stack(outputs)  # Stacks timesteps
        all_hiddens = torch.stack(layer_hiddens)  # type: ignore  # Stacks layers

        if self.batch_first:
            all_outputs = torch.transpose(all_outputs, 0, 1)

        return all_outputs, all_hiddens, total_ponder_cost


# torch.nn-style cell interface
class AdaptiveRNNCell(AdaptiveCellWrapper):
    """
    An adaptive-time variant of torch.nn.RNNCell.

    Parameters
    ----------
    input_size
        The number of expected features in the input.
    hidden_size
        The number of features in the hidden state.
    time_penalty
        How heavily to penalize the model for thinking too long. Tau in Graves 2016.
    bias
        Whether to use a learnable bias for the input-hidden and hidden-hidden
        functions.
    nonlinearity
        The nonlinearity to use. Can be either "tanh" or "relu".
    initial_halting_bias
        Value to initialize the halting unit's bias to. Recommended to set this
        to a negative number to prevent long ponder sequences early in training.
    ponder_epsilon
        When the halting values sum to more than 1 - ponder_epsilon, stop computation.
        Used to enable halting on the first step.
    time_limit
        Hard limit for how many substeps any computation can take. Intended to prevent
        overly-long computation early-on. M in Graves 2016.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_penalty: float,
        bias: bool = True,
        nonlinearity: str = "tanh",
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
    ):
        # Extra input for the first-step flag
        cell = torch.nn.RNNCell(input_size + 1, hidden_size, bias, nonlinearity)
        super().__init__(
            cell, time_penalty, initial_halting_bias, ponder_epsilon, time_limit
        )


class AdaptiveGRUCell(AdaptiveCellWrapper):
    """
    An adaptive-time variant of torch.nn.GRUCell.

    Parameters
    ----------
    input_size
        The number of expected features in the input.
    hidden_size
        The number of features in the hidden state.
    time_penalty
        How heavily to penalize the model for thinking too long. Tau in Graves 2016.
    bias
        Whether to use a learnable bias for the input-hidden and hidden-hidden
        functions.
    initial_halting_bias
        Value to initialize the halting unit's bias to. Recommended to set this
        to a negative number to prevent long ponder sequences early in training.
    ponder_epsilon
        When the halting values sum to more than 1 - ponder_epsilon, stop computation.
        Used to enable halting on the first step.
    time_limit
        Hard limit for how many substeps any computation can take. Intended to prevent
        overly-long computation early-on. M in Graves 2016.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_penalty: float,
        bias: bool = True,
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
    ):
        # Extra input for the first-step flag
        cell = torch.nn.GRUCell(input_size + 1, hidden_size, bias)
        super().__init__(
            cell, time_penalty, initial_halting_bias, ponder_epsilon, time_limit
        )


# torch.nn-style RNN interface
class AdaptiveRNN(_CellUnrollWrapper):
    """
    An adaptive-time variant of torch.nn.RNN.

    Parameters
    ----------
    input_size
        The number of expected features in the input.
    hidden_size
        The number of features in the hidden state.
    num_layers
        How many layers to use.
    time_penalty
        How heavily to penalize the model for thinking too long. Tau in Graves 2016.
    bias
        Whether to use a learnable bias for the input-hidden and hidden-hidden
        functions.
    nonlinearity
        The nonlinearity to use. Can be either "tanh" or "relu".
    initial_halting_bias
        Value to initialize the halting unit's bias to. Recommended to set this
        to a negative number to prevent long ponder sequences early in training.
    ponder_epsilon
        When the halting values sum to more than 1 - ponder_epsilon, stop computation.
        Used to enable halting on the first step.
    time_limit
        Hard limit for how many substeps any computation can take. Intended to prevent
        overly-long computation early-on. M in Graves 2016.
    batch_first
        If True, expects the first dimension of each sequence to be the batch axis
        and the second to be the sequence axis.
    dropout
        Amount of dropout to apply to the output of each layer except the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        time_penalty: float,
        bias: bool = True,
        nonlinearity: str = "tanh",
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        def make_cell(first):
            return AdaptiveRNNCell(
                input_size=input_size if first else hidden_size,
                hidden_size=hidden_size,
                time_penalty=time_penalty,
                bias=bias,
                nonlinearity=nonlinearity,
                initial_halting_bias=initial_halting_bias,
                ponder_epsilon=ponder_epsilon,
                time_limit=time_limit,
            )

        cells = [make_cell(True)] + [make_cell(False) for _ in range(num_layers - 1)]
        super().__init__(cells, batch_first, dropout)


class AdaptiveGRU(_CellUnrollWrapper):
    """
    An adaptive-time variant of torch.nn.GRU.

    Parameters
    ----------
    input_size
        The number of expected features in the input.
    hidden_size
        The number of features in the hidden state.
    num_layers
        How many layers to use.
    time_penalty
        How heavily to penalize the model for thinking too long. Tau in Graves 2016.
    bias
        Whether to use a learnable bias for the input-hidden and hidden-hidden
        functions.
    initial_halting_bias
        Value to initialize the halting unit's bias to. Recommended to set this
        to a negative number to prevent long ponder sequences early in training.
    ponder_epsilon
        When the halting values sum to more than 1 - ponder_epsilon, stop computation.
        Used to enable halting on the first step.
    time_limit
        Hard limit for how many substeps any computation can take. Intended to prevent
        overly-long computation early-on. M in Graves 2016.
    batch_first
        If True, expects the first dimension of each sequence to be the batch axis
        and the second to be the sequence axis.
    dropout
        Amount of dropout to apply to the output of each layer except the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        time_penalty: float,
        bias: bool = True,
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        def make_cell(first):
            return AdaptiveGRUCell(
                input_size=input_size if first else hidden_size,
                hidden_size=hidden_size,
                time_penalty=time_penalty,
                bias=bias,
                initial_halting_bias=initial_halting_bias,
                ponder_epsilon=ponder_epsilon,
                time_limit=time_limit,
            )

        cells = [make_cell(True)] + [make_cell(False) for _ in range(num_layers - 1)]
        super().__init__(cells, batch_first, dropout)



import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Tuple

class AdaptiveHaltingUnit(nn.Module):
    """
    Computes halting probabilities based on the model's hidden states.

    Parameters:
    -----------
    hidden_size : int
        The size of the hidden state.
    initial_bias : float
        Initial bias for the halting unit's linear layer.
    """

    def __init__(self, hidden_size: int, initial_bias: float = -1.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.linear.bias, initial_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probabilities.

        Parameters:
        -----------
        hidden : torch.Tensor
            The hidden state tensor.

        Returns:
        --------
        halting_prob : torch.Tensor
            Halting probabilities for each element in the batch.
        """
        logits = self.linear(hidden)
        halting_prob = self.sigmoid(logits)
        return halting_prob


class AdaptiveComputationTimeWrapper(nn.Module):
    """
    A generic wrapper to apply Adaptive Computation Time (ACT) to any model.

    Parameters:
    -----------
    model : nn.Module
        The model to wrap. It should accept inputs and hidden states, and return outputs and updated hidden states.
    compute_hidden : Callable
        A function that extracts a single hidden representation from the model's hidden states to compute halting probability.
    time_penalty : float
        Penalty term for the number of computation steps.
    initial_halting_bias : float
        Initial bias for the halting unit to encourage early halting.
    ponder_epsilon : float
        Epsilon to determine when to stop accumulating halting probabilities.
    time_limit : int
        Maximum number of computation steps per timestep.
    """

    def __init__(
        self,
        model: nn.Module,
        compute_hidden: Callable[[Any], torch.Tensor],
        time_penalty: float,
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
    ):
        super().__init__()
        self.model = model
        self.compute_hidden = compute_hidden  # Function to extract hidden for halting
        self.time_penalty = time_penalty
        self.ponder_epsilon = ponder_epsilon
        self.time_limit = time_limit

        # Initialize the halting unit based on the hidden size
        # Assume compute_hidden returns (batch_size, hidden_size)
        sample_hidden = self.compute_hidden(self._initialize_sample_hidden())
        hidden_size = sample_hidden.size(-1)
        self.halting_unit = AdaptiveHaltingUnit(hidden_size, initial_bias=initial_halting_bias)

    def _initialize_sample_hidden(self) -> Any:
        """
        Initialize a sample hidden state to infer hidden size.
        This should be overridden based on the specific model.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        """
        Forward pass with Adaptive Computation Time.

        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor of shape (batch, input_size).
        hidden : Any, optional
            Initial hidden state(s). The structure depends on the wrapped model.

        Returns:
        --------
        final_output : torch.Tensor
            The output after adaptive computation. Shape depends on the model.
        final_hidden : Any
            The final hidden state(s).
        ponder_cost : torch.Tensor
            Scalar tensor representing the ponder cost.
        ponder_steps : torch.Tensor
            Tensor indicating the number of computation steps for each batch element.
        """
        batch_size = inputs.size(0)
        device = inputs.device

        # Initialize halting probabilities and related tensors
        budget = torch.ones((batch_size, 1), device=device) - self.ponder_epsilon
        halting_sum = torch.zeros((batch_size, 1), device=device)
        halting_probs = []
        ponder_steps = torch.zeros((batch_size,), device=device)

        # Accumulate outputs weighted by halting probabilities
        accumulated_output = None  # To be defined based on model's output

        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self.model.init_hidden(batch_size)  # Assumes the model has an init_hidden method

        # Lists to store intermediate outputs
        accumulated_outputs = []
        accumulated_remainders = torch.zeros((batch_size, 1), device=device)

        for step in range(self.time_limit):
            # Forward pass through the model
            output, hidden = self.model(inputs, hidden)

            # Compute halting probability
            halting_prob = self.halting_unit(self.compute_hidden(hidden))  # Shape: (batch, 1)

            # Mask for elements that are still computing
            still_computing = (halting_sum < budget).float()

            # Mask for elements that will halt at this step
            new_halts = torch.clamp(budget - halting_sum, min=0.0)
            new_halts = torch.min(new_halts, halting_prob)

            # Update halting sum
            halting_sum = halting_sum + new_halts

            # Accumulate the output
            if accumulated_output is None:
                accumulated_output = new_halts * output
            else:
                accumulated_output = accumulated_output + new_halts * output

            # Accumulate the ponder steps
            ponder_steps = ponder_steps + (new_halts > 0).float().squeeze(-1)

            # Check if all elements have halted
            if halting_sum.ge(budget).all():
                break

        else:
            # For elements that didn't halt, use the last output
            remaining = budget - halting_sum
            remaining = torch.clamp(remaining, min=0.0)
            accumulated_output = accumulated_output + remaining * output
            ponder_steps = ponder_steps + 1

        # Compute ponder cost
        ponder_cost = self.time_penalty * ponder_steps.float().mean()

        return accumulated_output, hidden, ponder_cost, ponder_steps
