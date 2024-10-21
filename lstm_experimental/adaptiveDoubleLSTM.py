import torch
import torch.nn as nn
from typing import Optional, Tuple

class AdaptiveLSTMBlockWrapper(nn.Module):
    """
    Adaptive Computation Time (ACT) Wrapper for a Block of (LSTM1) -> (Activation) -> (LSTM2).

    This wrapper allows the computation block to adaptively determine the number of
    computation steps based on the input, potentially reducing computation for simpler inputs.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input to LSTM1.
    hidden_size : int
        The number of features in the hidden state for both LSTM1 and LSTM2.
    time_penalty : float
        How heavily to penalize the model for taking more computation steps. Tau in Graves 2016.
    activation_fn : nn.Module
        The activation function to apply between LSTM1 and LSTM2 (e.g., nn.ReLU()).
    initial_halting_bias : float, optional (default=-1.0)
        Initial bias for the halting unit to encourage early halting.
    ponder_epsilon : float, optional (default=1e-2)
        Epsilon to determine when to stop accumulating halting probabilities.
    time_limit : int, optional (default=100)
        Maximum number of computation steps per input to prevent excessive computation.
    bias : bool, optional (default=True)
        Whether to use a bias term in the LSTM layers.

    Attributes
    ----------
    l1 : nn.LSTMCell
        The first LSTM layer.
    activation : nn.Module
        The activation function applied between LSTM1 and LSTM2.
    l2 : nn.LSTMCell
        The second LSTM layer.
    halting_unit : nn.Sequential
        The halting unit comprising a linear layer followed by a sigmoid activation.
    time_penalty : float
        Penalty term for the number of computation steps.
    ponder_epsilon : float
        Epsilon threshold for halting.
    time_limit : int
        Maximum allowed computation steps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_penalty: float,
        activation_fn: nn.Module,
        initial_halting_bias: float = -1.0,
        ponder_epsilon: float = 1e-2,
        time_limit: int = 100,
        bias: bool = True,
    ):
        super().__init__()

        if time_penalty <= 0:
            raise ValueError("time_penalty must be positive.")
        if not (0.0 <= ponder_epsilon < 1.0):
            raise ValueError("ponder_epsilon must be in [0, 1).")

        self.time_penalty = time_penalty
        self.ponder_epsilon = ponder_epsilon
        self.time_limit = time_limit

        # First LSTM layer with an additional input for the first-step flag
        self.l1 = nn.LSTMCell(input_size + 1, hidden_size, bias=bias)

        # Activation function between LSTM1 and LSTM2
        self.activation = activation_fn

        # Second LSTM layer
        self.l2 = nn.LSTMCell(hidden_size, hidden_size, bias=bias)

        # Halting unit to compute halting probabilities
        self.halting_unit = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # Initialize the bias of the halting unit to encourage early halting
        nn.init.constant_(self.halting_unit[0].bias, initial_halting_bias)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass with Adaptive Computation Time.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, input_size).
        hidden : Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]], optional
            Initial hidden states for LSTM1 and LSTM2. If None, initialized to zeros.

        Returns
        -------
        accumulated_output : torch.Tensor
            The accumulated output after adaptive computation. Shape: (batch_size, hidden_size).
        final_hidden : Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            The final hidden states for LSTM1 and LSTM2.
        ponder_cost : torch.Tensor
            Scalar tensor representing the ponder cost for this input.
        ponder_steps : torch.Tensor
            Tensor indicating the number of computation steps taken for each batch element. Shape: (batch_size,).
        """
        batch_size, input_size = inputs.size()
        device = inputs.device

        # Initialize halting probability budget
        budget = torch.ones((batch_size, 1), device=device) - self.ponder_epsilon

        # Initialize accumulated halting probabilities
        halt_accum = torch.zeros((batch_size, 1), device=device)

        # Initialize accumulated output
        accumulated_output = torch.zeros((batch_size, self.l2.hidden_size), device=device)

        # Initialize accumulated remainder for ponder cost
        accumulated_remainder = torch.zeros_like(budget)

        # Initialize ponder steps
        ponder_steps = torch.zeros((batch_size,), device=device)

        # Initialize hidden states if not provided
        if hidden is None:
            h1 = torch.zeros((batch_size, self.l1.hidden_size), device=device)
            c1 = torch.zeros((batch_size, self.l1.hidden_size), device=device)
            h2 = torch.zeros((batch_size, self.l2.hidden_size), device=device)
            c2 = torch.zeros((batch_size, self.l2.hidden_size), device=device)
        else:
            (h1, c1), (h2, c2) = hidden

        # Initialize first-step flag
        first_step = True

        # Mask indicating which batch elements are still computing
        continuing_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)

        for step in range(self.time_limit):
            # Determine the first-step flag
            if first_step:
                flag = torch.ones((batch_size, 1), device=device)
                first_step = False
            else:
                flag = torch.zeros((batch_size, 1), device=device)

            # Concatenate input with the first-step flag
            step_input = torch.cat([inputs, flag], dim=1)  # Shape: (batch_size, input_size + 1)

            # Forward pass through LSTM1
            h1, c1 = self.l1(step_input, (h1, c1))  # Each of shape: (batch_size, hidden_size)

            # Apply activation function
            activated = self.activation(h1)  # Shape: (batch_size, hidden_size)

            # Forward pass through LSTM2
            h2, c2 = self.l2(activated, (h2, c2))  # Each of shape: (batch_size, hidden_size)

            # Compute halting probability using the halting unit
            step_halt = self.halting_unit(h2)  # Shape: (batch_size, 1)

            # Mask halting probabilities for continuing elements
            masked_halt = step_halt * continuing_mask.float()  # Shape: (batch_size, 1)

            # Accumulate halting probabilities
            halt_accum = halt_accum + masked_halt  # Shape: (batch_size, 1)

            # Determine which elements should halt
            ending_mask = (halt_accum > budget) & continuing_mask  # Shape: (batch_size, 1)

            # Update continuing mask
            continuing_mask = continuing_mask & ~ending_mask  # Shape: (batch_size, 1)

            # Compute masked remainder for halting elements
            masked_remainder = ending_mask.float() * (1 - halt_accum)  # Shape: (batch_size, 1)

            # Combine masked halting probabilities and remainders
            combined_mask = masked_halt + masked_remainder  # Shape: (batch_size, 1)

            # Accumulate the weighted outputs
            accumulated_output = accumulated_output + (combined_mask * h2)  # Shape: (batch_size, hidden_size)

            # Accumulate the halting probabilities for ponder cost
            accumulated_remainder = accumulated_remainder + masked_halt  # Shape: (batch_size, 1)

            # Accumulate the number of steps taken
            ponder_steps = ponder_steps + continuing_mask.squeeze(1).float()  # Shape: (batch_size,)

            # Early stopping if all elements have halted
            if not continuing_mask.any():
                break

        else:
            # Handle elements that did not halt within the time limit
            masked_remainder = continuing_mask.float() * (1 - halt_accum)  # Shape: (batch_size, 1)
            accumulated_output = accumulated_output + (masked_remainder * h2)  # Shape: (batch_size, hidden_size)

        # Compute ponder cost (negative to maximize halting probabilities)
        ponder_cost = -1.0 * self.time_penalty * accumulated_remainder.mean()

        # Final hidden states
        final_hidden = ((h1, c1), (h2, c2))

        return accumulated_output, final_hidden, ponder_cost, ponder_steps
