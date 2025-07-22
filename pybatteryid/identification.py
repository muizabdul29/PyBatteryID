"""Utilities related to model identification."""


from typing import Literal

import numpy as np

from .modelstructure import ModelStructure
from .dataclasses import Model
from .typeddicts import CurrentVoltageData
from .regression import setup_regression, combine_regression_problems, run_optimizer
from .basisfunctions import combine_symbols, generate_signals, generate_basis_function_signals, \
    generate_signal_trajectories


# pylint: disable=R0914,W0102
def setup_regression_problems(datasets: list[CurrentVoltageData],
                              model_structure: ModelStructure,
                              model_order: int, nonlinearity_order: int,
                              input_symbols: list[str] = ['i']):
    """Generate regression matrices for the given dataset(s)."""
    # Shorthand for convenience
    ms = model_structure
    #
    regression_problems = []
    for dataset in datasets:
        signals = generate_signals(dataset, ms.battery_capacity, ms.sampling_period,
                                   ms.emf_function, ms.hysteresis_function)
        #
        bf_signal_vector = generate_basis_function_signals(ms.basis_functions, signals)
        hysteresis_bf_signal_vector = generate_basis_function_signals(ms.hysteresis_basis_functions,
                                                                      signals)
        #
        io_signals = [signals.find(s)
                      for s in ['v'] + input_symbols + (['h'] if ms.hysteresis_function else [])]
        #
        signal_tuple = io_signals, bf_signal_vector.signals, hysteresis_bf_signal_vector.signals
        signal_trajectories = generate_signal_trajectories(signal_tuple,
                                                            model_order, model_order)
        io_trajectories, p_trajectories, h_trajectories = signal_trajectories
        #
        basis_function_keys = combine_symbols(bf_signal_vector.get_symbols(), nonlinearity_order)
        hysteresis_basis_function_keys = hysteresis_bf_signal_vector.get_symbols()
        #
        regression_matrix, regressor_labels = setup_regression(io_trajectories,
                                                                p_trajectories,
                                                                h_trajectories,
                                                                basis_function_keys,
                                                                hysteresis_basis_function_keys)
        output_vector = np.array(signals.find('v').trajectory[model_order:])
        #
        regression_problems.append( (regression_matrix, output_vector) )
    #
    return regression_problems, regressor_labels


# pylint: disable=too-many-arguments, R0917
def identify_model(datasets: list[CurrentVoltageData] | CurrentVoltageData,
                   model_structure: ModelStructure,
                   model_order: int, nonlinearity_order: int,
                   optimizers: Literal['lasso.cvxopt', 'lasso.sklearn', 'ridge.sklearn'],
                   combining_strategy: Literal['concatenate', 'interleave']='interleave'):
    """Identify a battery model using the provided identification
    dataset along with the desired model order and basis-function
    complexity."""
    # Shorthand for convenience
    ms = model_structure
    #
    if not isinstance(datasets, list):
        datasets = [datasets]
    regression_problems, regressor_labels = setup_regression_problems(datasets,
                                                                      model_structure,
                                                                      model_order,
                                                                      nonlinearity_order)
    # Combine multiple regression problems
    regression_matrix, output_vector = combine_regression_problems(regression_problems,
                                                                    strategy=combining_strategy)
    #
    if not optimizers:
        raise ValueError('Unspecified optimization routine(s).')
    #
    model_estimate = np.array([])
    for optimizer in optimizers:
        #
        model_estimate = run_optimizer(regression_matrix, output_vector, optimizer.lower())
        #
        if 'lasso' in optimizer.lower():
            selected_regressors = np.abs(model_estimate) > 1e-5
            regression_matrix = regression_matrix[:, selected_regressors]  # type: ignore
            regressor_labels = np.array(regressor_labels)[selected_regressors]
            model_estimate = np.array(model_estimate)[selected_regressors]

    return Model(ms.battery_capacity,
                 ms.sampling_period,
                 model_order,
                 nonlinearity_order,
                 regressor_labels,
                 model_estimate,
                 ms.basis_functions,
                 ms.hysteresis_basis_functions,
                 ms.emf_function,
                 ms.hysteresis_function)
