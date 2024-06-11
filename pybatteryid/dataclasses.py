"""
A collection of classes (@dataclasses) for various model
features.
"""

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

@dataclass
class VoltageFunction:
    """A callable dataclass for creating voltage function."""
    func: Callable
    soc_values: list[float]

    def __call__(self, soc):
        return self.func(soc)

    def get_soc_values(self) -> list[float]:
        """Get the soc values used to construct the model."""
        return self.soc_values

@dataclass
class Model:
    """Represents a battery model."""
    model_order: int
    nonlinearity_order: int
    model_terms: np.ndarray[Any, Any]
    model_estimate: np.ndarray[Any, Any]

@dataclass
class BasisFunction:
    """Represents a basis function."""
    variable: str
    operation: Any
    arguments: list[Any]
    function_string: str

@dataclass
class Signal:
    """A realization of a signal."""
    symbol: str
    trajectory: list[float]
    equation: Callable[..., float]
    equation_order: float = field(default_factory=lambda: 0)

    def update_trajectory(self, *args):
        """Add a new value to the corresponding
        trajectory."""

        # Add past values to the arguments depending on the
        # order of the difference equation
        past_values = self.trajectory[-self.equation_order:] if self.equation_order > 0 else []
        updated_args = tuple(past_values) + args

        self.trajectory.append( self.equation( *updated_args ) )

@dataclass
class SignalVector:
    """A collection of signals."""
    signals: list[Signal]

    def get_symbols(self):
        """Get symbols."""
        symbols = []
        for signal in self.signals:
            symbols.append( signal.symbol )
        return symbols

    def add(self, signal):
        """Add a signal to the collection."""
        self.signals.append(signal)

    def find(self, symbol):
        """Find a signal using its symbol."""
        result = None
        for signal in self.signals:
            if signal.symbol == symbol:
                result = signal
                break
        if result is None:
            raise ValueError('Signal not found.')
        return result
