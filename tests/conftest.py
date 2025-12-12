"""Configure tests."""

import os
import numpy as np
import pytest

from pybatteryid.modelstructure import ModelStructure


@pytest.fixture(scope="session")
def model_structure():
    """Set up an example model structure."""
    #
    battery_capacity = 10344.169
    sampling_period = 1
    parent_directory = os.path.dirname(os.path.dirname(__file__))
    #
    emf_data = np.load(f'{parent_directory}/examples/data/'
                       'nmc_with_temperature/gitt_data.npy',
                       allow_pickle=True).item()
    #
    ms = ModelStructure(battery_capacity, sampling_period)
    ms.add_emf_function({'soc_values': emf_data['soc_values'],
                                      'voltage_values': emf_data['voltage_values'],
                                      'dVdT_values': emf_data['dVoltage_dTemperature_values'],
                                      'reference_temperature_value': emf_data['reference_temperature_value']}) # pylint: disable=C0301
    #
    yield ms


@pytest.fixture(scope="session")
def dataset():
    """Load an example identification dataset. """
    #
    parent_directory = os.path.dirname(os.path.dirname(__file__))
    ds = np.load(f'{parent_directory}/examples/data/'
                      'nmc_with_temperature/identification_data_2_6.npy',
                      allow_pickle=True).item()
    ds['initial_soc'] = 0.9979013241720326
    return ds
