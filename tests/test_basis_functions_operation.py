"""
Testing related to basis functions operation from
user-provided strings.
"""

import math as m

from pybatteryid.basisfunctions import generate_signals, extract_basis_functions, \
    generate_basis_function_signals
from pybatteryid.modelstructure import ModelStructure
from pybatteryid.typeddicts import CurrentVoltageData



def test_no_operation(model_structure: ModelStructure, dataset: CurrentVoltageData):
    """No operation."""

    scheduling_signals = generate_signals(dataset,
                                          model_structure.battery_capacity,
                                          model_structure.sampling_period,
                                          model_structure.emf_function, None)
    s = scheduling_signals.find('s')
    #
    strings = ['s']
    basis_functions = extract_basis_functions(strings)
    signal_vector = generate_basis_function_signals(basis_functions, scheduling_signals)
    #
    result = signal_vector.find(strings[0])

    assert result.symbol == s.symbol
    assert result.trajectory == s.trajectory

def test_inverse_operation(model_structure: ModelStructure, dataset: CurrentVoltageData):
    """Inverse operation."""

    scheduling_signals = generate_signals(dataset,
                                        model_structure.battery_capacity,
                                        model_structure.sampling_period,
                                        model_structure.emf_function, None)
    s = scheduling_signals.find('s')
    #
    strings = ['1/s']
    basis_functions = extract_basis_functions(strings)
    signal_vector = generate_basis_function_signals(basis_functions, scheduling_signals)
    #
    result = signal_vector.find(strings[0])

    assert result.symbol == strings[0]
    assert result.trajectory == [1 / x for x in s.trajectory]

def test_log_operation(model_structure: ModelStructure, dataset: CurrentVoltageData):
    """Logarithmic operation."""

    scheduling_signals = generate_signals(dataset,
                                          model_structure.battery_capacity,
                                          model_structure.sampling_period,
                                          model_structure.emf_function, None)
    s = scheduling_signals.find('s')
    #
    strings = ['log[s]']
    basis_functions = extract_basis_functions(strings)
    signal_vector = generate_basis_function_signals(basis_functions, scheduling_signals)
    #
    result = signal_vector.find(strings[0])

    assert result.symbol == strings[0]
    assert result.trajectory == [m.log(x) for x in s.trajectory]

def test_exp_sqrt_abs_operation(model_structure: ModelStructure, dataset: CurrentVoltageData):
    """Exponential square-root operation."""

    scheduling_signals = generate_signals(dataset,
                                          model_structure.battery_capacity,
                                          model_structure.sampling_period,
                                          model_structure.emf_function, None)
    i = scheduling_signals.find('i')
    s = scheduling_signals.find('s')
    #
    strings = ['exp[0.05*sqrt[|i|]]', 'exp[-2.759*sqrt[0.5*s+140]]']
    basis_functions = extract_basis_functions(strings)
    signal_vector = generate_basis_function_signals(basis_functions, scheduling_signals)
    #
    result = signal_vector.find(strings[0])

    assert result.symbol == strings[0]
    assert result.trajectory == [m.exp(0.05*m.sqrt(abs(x))) for x in i.trajectory]

    result = signal_vector.find(strings[1])

    assert result.symbol == strings[1]
    assert result.trajectory == [m.exp(-2.759*m.sqrt(0.5*x+140)) for x in s.trajectory]

def test_exp_pow_abs_operation(model_structure: ModelStructure, dataset: CurrentVoltageData):
    """Exponential arbitrary power operation."""

    scheduling_signals = generate_signals(dataset,
                                          model_structure.battery_capacity,
                                          model_structure.sampling_period,
                                          model_structure.emf_function, None)
    T = scheduling_signals.find('T') # pylint: disable=C0103
    i = scheduling_signals.find('i')
    #
    strings = ['exp[[0.00366*T+1]^-1]', 'exp[-2.759*[|-1.5*i+1890|]^+2.89]']
    basis_functions = extract_basis_functions(strings)
    signal_vector = generate_basis_function_signals(basis_functions, scheduling_signals)
    #
    result = signal_vector.find(strings[0])

    assert result.symbol == strings[0]
    assert result.trajectory == [m.exp((0.00366*x+1)**-1) for x in T.trajectory]

    result = signal_vector.find(strings[1])

    assert result.symbol == strings[1]
    assert result.trajectory == [m.exp(-2.759*(-1.5*x+1890)**2.89) for x in i.trajectory]
