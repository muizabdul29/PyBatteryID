"""Configure tests."""

import os
import numpy as np
import pytest

from pybatteryid.modelstructure import ModelStructure
from pybatteryid.basisfunctions import generate_signals


@pytest.fixture(scope="session")
def scheduling_signals():
    """Setup model structure and generate scheduling signals
    for the example dataset."""
    #
    battery_capacity = 10344.169
    sampling_period = 1
    parent_directory = os.path.dirname(os.path.dirname(__file__))
    #
    dataset = np.load(f'{parent_directory}/examples/data/'
                      'nmc_with_temperature/identification_data_2_6.npy',
                      allow_pickle=True).item()
    dataset['initial_soc'] = 0.9979013241720326
    #
    emf_data = np.load(f'{parent_directory}/examples/data/'
                       'nmc_with_temperature/gitt_data.npy',
                       allow_pickle=True).item()
    #
    model_structure = ModelStructure(battery_capacity, sampling_period)
    model_structure.add_emf_function({'soc_values': emf_data['soc_values'],
                                      'voltage_values': emf_data['voltage_values'],
                                      'dVdT_values': emf_data['dVoltage_dTemperature_values'],
                                      'reference_temperature_value': emf_data['reference_temperature_value']}) # pylint: disable=C0301
    #
    signals = generate_signals(dataset,
                               battery_capacity,
                               sampling_period,
                               model_structure._emf_function, # pylint: disable=W0212
                               None)
    yield signals
