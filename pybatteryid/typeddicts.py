"""
A collection of classes for various typed dictionaries.
"""

from typing import TypedDict
from numpy.typing import NDArray

class VoltageSocData(TypedDict):
    """A dictionary for voltage--soc data."""
    soc_values: NDArray
    voltage_values: NDArray

class CurrentVoltageData(TypedDict):
    """A dictionary for current--voltage data."""
    time_values: NDArray
    current_values: NDArray
    voltage_values: NDArray
