"""
This file contains the main function(s) to construct
a voltage (e.g., EMF) model.
"""


import numpy as np
from scipy.interpolate import PchipInterpolator

from .dataclasses import VoltageFunction
from .typeddicts import VoltageSocData


def create_soc_interpolator(soc_values: np.ndarray, y_values: np.ndarray,
                            extrapolate=False) -> PchipInterpolator:
    """Create interpolation function from y--soc data. Note that
    y can be either voltage or temperature."""

    # If the SOC direction is descending, we sort it to
    # ascending first.
    if any(np.diff(soc_values) < 0):
        # We reverse the lists
        soc_values = soc_values[::-1]
        y_values = y_values[::-1]

    interpolator = PchipInterpolator(soc_values, y_values, extrapolate=extrapolate)
    return interpolator


def load_voltage_model(dataset: VoltageSocData):
    """Load temperature-dependent voltage model."""
    #
    dVdT_soc_function = None # pylint: disable=C0103
    reference_temperature = None
    if 'dVdT_values' in dataset:
        if 'reference_temperature_value' not in dataset:
            raise ValueError('Reference temperature value not specified.')
        # pylint: disable=C0103
        dVdT_soc_function = create_soc_interpolator(dataset['soc_values'], dataset['dVdT_values'])
        reference_temperature = dataset['reference_temperature_value']

    # We create and store interpolation functions for
    # the given Voltage--SOC data.
    voltage_soc_function = create_soc_interpolator(dataset['soc_values'], dataset['voltage_values'])

    return VoltageFunction(voltage_soc_function, dVdT_soc_function, reference_temperature)
