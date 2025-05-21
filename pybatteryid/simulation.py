"""Voltage simulation utilities."""


import numpy as np

from .dataclasses import Model, Signal
from .typeddicts import CurrentVoltageData
from .basisfunctions import generate_signals, generate_basis_function_signals, \
    generate_signal_trajectories


# pylint: disable=too-many-locals
def simulate_model(model: Model, dataset: CurrentVoltageData):
    """Simulate the battery voltage using the provided model
    with the battery current as input."""
    #
    no_of_initial_values = len(dataset['voltage_values'])
    # Check if minimum number of initial values are provided.
    if no_of_initial_values < model.model_order:
        raise ValueError(f'At least {model.model_order} initial voltage value(s)'
                            ' should be provided.')
    #
    signal_vector = generate_signals(dataset, model.battery_capacity, model.sampling_period,
                                     model.emf_function, model.hysteresis_function)
    #
    bf_signal_vector = generate_basis_function_signals(model.basis_functions, signal_vector)
    hysteresis_bf_signal_vector = generate_basis_function_signals(model.hysteresis_basis_functions,
                                                                  signal_vector)
    #
    io_signals = [ signal_vector.find('v'), signal_vector.find('i') ]
    if model.hysteresis_function is not None:
        io_signals.append( signal_vector.find('h') )
    #
    io_signals[0].trajectory.extend([np.nan] *
                                    (len(dataset['current_values']) - no_of_initial_values))
    #
    signal_tuple = io_signals, bf_signal_vector.signals, hysteresis_bf_signal_vector.signals
    signal_trajectories = generate_signal_trajectories(signal_tuple, model.model_order,
                                                        no_of_initial_values)
    io_trajectories, p_trajectories, h_trajectories = signal_trajectories
    #
    simulated_voltage = Signal('v_sim', signal_vector.find('v').trajectory[:no_of_initial_values],
                               lambda x: x)
    combined_trajectories_dict = io_trajectories | p_trajectories | h_trajectories
    #
    for delay in range(1, model.model_order + 1):
        key = simulated_voltage.symbol + ('(k-' + str(delay) + ')' if delay > 0 else '(k)')
        #
        combined_trajectories_dict[key] = [simulated_voltage.trajectory[-delay]]

    for k in range(len(signal_vector.find('i').trajectory) - no_of_initial_values):
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

    emf_trajectory = []
    for i, s in enumerate(signal_vector.find('s').trajectory):
        t = signal_vector.find('T').trajectory[i] if 'temperature_values' in dataset else None
        emf_trajectory.append(model.emf_function(s, t))
    #
    return np.add(emf_trajectory, simulated_voltage.trajectory)
