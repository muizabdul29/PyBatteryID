"""
Contains the class `ModelStructure`.
"""

from typing import Literal

import numpy as np

from .voltage import load_voltage_model
from .basisfunctions import extract_basis_functions, combine_symbols, \
    generate_signals, generate_basis_functions, generate_signal_trajectories
from .regression import setup_regression, run_optimizer
from .dataclasses import VoltageFunction, BasisFunction, Model, Signal
from .typeddicts import VoltageSocData, CurrentVoltageData

class ModelStructure:
    """This class allows battery model identification and simulation employing
    the proposed model structure using the input-output (IO) representation in
    the linear parameter-varying (LPV) framework.

    Attributes
    ----------
    battery_capacity : float
        Capacity of the battery being modelled.
    sampling_period: float
        Model sampling time.
    """

    battery_capacity: float
    sampling_period: float

    _emf_function: VoltageFunction
    _hysteresis_function: VoltageFunction | None

    _basis_functions: list[BasisFunction]
    _hysteresis_basis_functions: list[BasisFunction]

    def __init__(self, battery_capacity: float, sampling_period: float):
        self.battery_capacity = battery_capacity
        self.sampling_period = sampling_period

        self._hysteresis_function = None

        self._basis_functions = []
        self._hysteresis_basis_functions = []

    def add_emf_function(self, voltage_soc_data: VoltageSocData):
        """Add EMF function used to decompose battery voltage into
        overpotentials and vice versa."""
        self._emf_function = load_voltage_model(voltage_soc_data['soc_values'],
                                                voltage_soc_data['voltage_values'])

    def add_hysteresis_function(self, voltage_soc_data: VoltageSocData):
        """Add hysteresis function to be used as second model input."""
        self._hysteresis_function = load_voltage_model(voltage_soc_data['soc_values'],
                                                       voltage_soc_data['voltage_values'])

    def add_basis_functions(self, basis_function_strings: list[str],
                            hysteresis_basis_function_strings: list[str] | None = None):
        """Add basis functions by extracting signal variable, the desired
        operation and its arguments."""
        self._basis_functions = extract_basis_functions(basis_function_strings)
        if hysteresis_basis_function_strings is not None:
            hbfs = hysteresis_basis_function_strings
            self._hysteresis_basis_functions = extract_basis_functions(hbfs)

    # pylint: disable=too-many-locals
    def simulate(self, model: Model, dataset: CurrentVoltageData, initial_soc: float):
        """Simulate the battery voltage using the provided model
        with the battery current as input."""
        #
        no_of_initial_values = len(dataset['voltage_values'])
        # Check if minimum number of initial values are provided.
        if no_of_initial_values < model.model_order:
            raise ValueError(f'At least {model.model_order} initial voltage value(s)'
                             ' should be provided.')
        #
        signals = generate_signals(dataset, self._emf_function, self._hysteresis_function,
                                   initial_soc, self.sampling_period, self.battery_capacity)
        #
        basis_functions = generate_basis_functions(self._basis_functions, signals)
        hysteresis_basis_functions = generate_basis_functions(self._hysteresis_basis_functions,
                                                              signals)
        #
        io_signals = [ signals.find('v'), signals.find('i') ]
        if self._hysteresis_function is not None:
            io_signals.append( signals.find('h') )
        #
        io_signals[0].trajectory.extend([np.nan] *
                                        (len(dataset['current_values']) - no_of_initial_values))
        #
        signal_tuple = io_signals, basis_functions.signals, hysteresis_basis_functions.signals
        signal_trajectories = generate_signal_trajectories(signal_tuple, model.model_order,
                                                           no_of_initial_values)
        io_trajectories, p_trajectories, h_trajectories = signal_trajectories
        #
        simulated_voltage = Signal('v_sim', signals.find('v').trajectory[:no_of_initial_values],
                                   lambda x: x)
        combined_trajectories_dict = io_trajectories | p_trajectories | h_trajectories
        #
        for delay in range(1, model.model_order + 1):
            key = simulated_voltage.symbol + ('(k-' + str(delay) + ')' if delay > 0 else '(k)')
            #
            combined_trajectories_dict[key] = [simulated_voltage.trajectory[-delay]]

        for k in range(len(signals.find('i').trajectory) - no_of_initial_values):
            #
            phi = []
            for key in model.model_terms:
                # Replace v with v_sim in the symbols
                new_symbols = []
                for sym in key.split('×'):
                    sym_parts = sym.split('(')
                    if sym_parts[0] == 'v':
                        new_symbols.append(simulated_voltage.symbol + '(' + sym_parts[1])
                    else:
                        new_symbols.append(sym)
                # The value corresponding to each column for kth time instant.
                term_value = np.prod([combined_trajectories_dict[sym][k] for sym in new_symbols])
                phi.append(term_value)

            # Newly calculated voltage value at kth time instant.
            v_sim_new_value = np.dot( phi, model.model_estimate )
            # Make updates.
            simulated_voltage.update_trajectory(v_sim_new_value)
            for delay in range(1, model.model_order + 1):
                key = simulated_voltage.symbol + ('(k-' + str(delay) + ')' if delay > 0 else '(k)')
                #
                combined_trajectories_dict[key].append( simulated_voltage.trajectory[-delay] )

        emf_trajectory = self._emf_function(signals.find('s').trajectory)
        return np.add(emf_trajectory, simulated_voltage.trajectory)

    # pylint: disable=too-many-arguments
    def identify(self, dataset: CurrentVoltageData, initial_soc: float,
                 model_order: int, nonlinearity_order: int, optimizers: Literal['lasso', 'ridge']):
        """Identify a battery model using the provided identification
        dataset along with the desired model order and basis-function
        complexity."""
        #
        signals = generate_signals(dataset, self._emf_function, self._hysteresis_function,
                                   initial_soc, self.sampling_period, self.battery_capacity)
        #
        basis_functions = generate_basis_functions(self._basis_functions, signals)
        hysteresis_basis_functions = generate_basis_functions(self._hysteresis_basis_functions,
                                                              signals)
        #
        io_signals = [ signals.find('v'), signals.find('i') ]
        if self._hysteresis_function is not None:
            io_signals.append( signals.find('h') )
        #
        signal_tuple = io_signals, basis_functions.signals, hysteresis_basis_functions.signals
        signal_trajectories = generate_signal_trajectories(signal_tuple, model_order, model_order)
        io_trajectories, p_trajectories, h_trajectories = signal_trajectories
        #
        basis_function_keys = combine_symbols(basis_functions.get_symbols(), nonlinearity_order)
        hysteresis_basis_function_keys = hysteresis_basis_functions.get_symbols()
        #
        regression_matrix, regressor_labels = setup_regression(io_trajectories,
                                                               p_trajectories,
                                                               h_trajectories,
                                                               basis_function_keys,
                                                               hysteresis_basis_function_keys)
        output_vector = np.array(signals.find('v').trajectory[model_order:])
        #
        if not optimizers:
            raise ValueError('Unspecified optimization routine(s).')
        #
        model_estimate = np.array([])
        for optimizer in optimizers:
            #
            model_estimate = run_optimizer(regression_matrix, output_vector, optimizer)
            #
            if 'lasso' in optimizer:
                selected_regressors = np.abs(model_estimate) > 1e-5
                regression_matrix = regression_matrix[:, selected_regressors] # type: ignore
                regressor_labels = np.array(regressor_labels)[selected_regressors]

        return Model(model_order, nonlinearity_order, regressor_labels, model_estimate)
