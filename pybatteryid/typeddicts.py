"""
A collection of classes for various typed dictionaries.
"""


from typing import TypedDict, NotRequired
from numpy.typing import NDArray


class VoltageSocData(TypedDict):
    """A dictionary for voltage--soc data."""
    soc_values: NDArray
    voltage_values: NDArray
    dVdT_values: NotRequired[NDArray]
    reference_temperature_value: NotRequired[float]


class CurrentVoltageData(TypedDict):
    """A dictionary for current--voltage data."""
    initial_soc: float
    time_values: NDArray
    current_values: NDArray
    voltage_values: NDArray
    temperature_values: NotRequired[NDArray]
