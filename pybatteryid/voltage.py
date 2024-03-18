"""
This file contains the main function(s) to construct
a voltage (e.g., EMF) model.
"""

import numpy as np

from scipy.interpolate import interp1d, PchipInterpolator

from .dataclasses import VoltageFunction

def load_voltage_model(
        soc_values: list[float], voltage_values: list[float],
        extrapolate=False, interpolator='pchip'
    ) -> VoltageFunction:
    """Load the model directly from the (saved) data."""

    # If the SOC direction is descending, we sort it to
    # ascending first.
    if any(np.diff(soc_values) < 0):
        # We reverse the lists
        soc_values = list(soc_values[::-1])
        voltage_values = list(voltage_values[::-1])

    if interpolator == '1d':
        fill_value = 'extrapolate' if extrapolate else np.nan
        bounds_error = not extrapolate
        interpolator = interp1d(
            soc_values, voltage_values, bounds_error=bounds_error,
            fill_value=fill_value, # type: ignore
            assume_sorted=True
        )
    else:
        interpolator = PchipInterpolator(soc_values, voltage_values, extrapolate=extrapolate)

    return VoltageFunction(
        interpolator,
        soc_values
    )
