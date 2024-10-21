import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Tuple

from models import AdaptiveComputationTimeWrapper


class AdaptiveLSTMWrapper(AdaptiveComputationTimeWrapper):
    """
    Adaptive Computation Time wrapper for LSTM models.
    """

    def _initialize_sample_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize sample hidden and cell states.
        """
        # Assuming hidden_size is accessible via the model
        hidden_size = self.model.hidden_size
        batch_size = 1
        h_0 = torch.zeros((batch_size, hidden_size))
        c_0 = torch.zeros((batch_size, hidden_size))
        return (h_0, c_0)

    def compute_hidden(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Extract the hidden state to compute halting probability.
        """
        h_t, _ = hidden
        return h_t  # Shape: (batch, hidden_size)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Override forward to handle LSTM's hidden and cell states.
        """
        return super().forward(inputs, hidden)
