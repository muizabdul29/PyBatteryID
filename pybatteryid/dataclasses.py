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
    emf_func: Callable
    dvdt_func: Callable | None
    reference_temperature: float | None

    def __call__(self, soc: float, temperature: float | None = None):
        voltage_value = self.emf_func(soc)
        #
        if temperature is not None:
            if self.dvdt_func is None and self.reference_temperature is None:
                return voltage_value
            #
            if self.dvdt_func is None:
                raise ValueError('Temperature-dependance of the voltage is not specified.')
            if self.reference_temperature is None:
                raise ValueError('Reference temperature value not specified.')
            #
            dvdt_value = self.dvdt_func(soc)
            return voltage_value + dvdt_value * (temperature - self.reference_temperature)

        return voltage_value


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


# pylint: disable=R0902
@dataclass
class Model:
    """Represents a battery model."""
    battery_capacity: float
    sampling_period: float
    model_order: int
    nonlinearity_order: int
    model_terms: np.ndarray[Any, Any]
    model_estimate: np.ndarray[Any, Any]
    basis_functions: list[BasisFunction]
    hysteresis_basis_functions: list[BasisFunction]
    emf_function: VoltageFunction
    hysteresis_function: VoltageFunction | None
