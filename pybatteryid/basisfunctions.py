"""Utilities concerning basis functions."""

import re
import math as m
from enum import Enum
from typing import Tuple, Any
from itertools import combinations_with_replacement

import numpy as np
from .dataclasses import BasisFunction, Signal, SignalVector, VoltageFunction
from .typeddicts import CurrentVoltageData

class Operation(Enum):
    """List of allowed operations."""
    NONE = 0
    INVERSE = 1
    LOGARITHM = 2
    LOWPASS = 3
    EXP_SQRT_ABS = 4
    EXP_POWER_ABS = 5

def extract_basis_functions(basis_function_strings: list[str]) -> list[BasisFunction]:
    """Extract basis functions from a list of strings by recognizing
    symbols of various signals and respective operations."""
    basis_functions = []
    # We define tuples for various operations in the format
    # (regex, indices for variable and arguments, operation).
    identifiers = [
        (r"^(s|i|d|v|T)$", [0], Operation.NONE),
        (r"^1/(s|d|i|v|T)$", [0], Operation.INVERSE),
        (r"^log\[(s|d|i|v|T)\]$", [0], Operation.LOGARITHM),
        (r"^exp\[(((\+|-)?([0-9]*[.])?[0-9]+)\*)?(sqrt)\[(\|)?"
         r"(((\+|-)?([0-9]*[.])?[0-9]+)\*)?(s|d|i|v|T)([+-](([0-9]*[.])?[0-9]+))?"
         r"(\|)?\]\]$", [10, 1, 7, 11, 5, 14, 4], Operation.EXP_SQRT_ABS),
        (r"^exp\[(((\+|-)?([0-9]*[.])?[0-9]+)\*)?"
         r"\[(\|)?(((\+|-)?([0-9]*[.])?[0-9]+)\*)?(s|d|i|v|T)([+-](([0-9]*[.])?[0-9]+))?(\|)?\]"
         r"(\^(([0-9]*[.])?[0-9]+))?\]$", [9, 1, 6, 10, 4, 13, 15], Operation.EXP_POWER_ABS),
        (r"^(s|i|d|v|T)\[(([0-9]*[.])?[0-9]+),"
         r"(([0-9]*[.])?[0-9]+)\]$", [0, 1, 3], Operation.LOWPASS)
    ]
    #
    for i, function_string in enumerate(basis_function_strings):
        for identifier, indices, operation in identifiers:
            #
            result = re.findall(identifier, function_string.strip())
            if len(result) > 0:
                result = result[0] if isinstance(result[0], tuple) else result
                args = [ result[i] for i in indices[1:] ]
                basis_functions.append(BasisFunction(result[indices[0]], operation, args,
                                                     function_string))
                break
        if len(basis_functions) == i:
            raise ValueError(f"Could not recognize expression: {function_string}")
    return basis_functions

def perform_signal_operation(signal: Signal, operation: Operation, args: list[Any],
                             function_string: str) -> Signal:
    """Perform an operation on a signal."""
    result_signal = signal
    if operation == Operation.INVERSE:
        result_signal = Signal(function_string,
                               [1 / x for x in signal.trajectory],
                               lambda x: 1 / x)
    elif operation == Operation.LOGARITHM:
        result_signal = Signal(function_string,
                               [m.log(x) for x in signal.trajectory],
                               m.log)
    elif operation in (Operation.EXP_SQRT_ABS, Operation.EXP_POWER_ABS):
        def equation(x):
            gamma_1 = float(args[0]) if args[0] != '' else 1
            gamma_2 = float(args[1]) if args[1] != '' else 1
            gamma_3 = float(args[2]) if args[2] != '' else 0
            result = gamma_2 * x + gamma_3
            if args[3] == '|' and args[4] == '|':
                result = abs(result)
            if args[5] == 'sqrt':
                result = m.sqrt(result)
            else:
                exponent = float(args[5]) if args[5] != '' else 1
                result = result**exponent
            return m.exp(gamma_1 * result)
        #
        result_signal = Signal(function_string,
                               [equation(x) for x in signal.trajectory],
                               equation)
    elif operation == Operation.LOWPASS:
        epsilon = lambda d: float(args[1]) if d == 0 else float(args[0]) # pylint: disable=C3001
        #
        result_signal = Signal(function_string, [signal.trajectory[0]],
                               lambda past_value, d: epsilon(d) * past_value + (1 - epsilon(d)) * d,
                               equation_order=1)
        # Update the trajectory
        for value in signal.trajectory[1:]:
            result_signal.update_trajectory(value)

    return result_signal

def combine_symbols(symbols: list[str], nl_order: int):
    """Use symbols to generate higher-order combinations."""
    #
    combinations = []
    for number_of_terms in range(1, nl_order + 1):
        combinations += list(combinations_with_replacement(symbols, number_of_terms))

    # We develop the conditions to filter unwanted combinations.
    # 1. If a key has more than one delta symbols
    def more_than_one_delta(key):
        return [ k[0] == 'd' for k in key ].count(True) > 1
    # 2. If a key has `s` and its inverse.
    def s_and_inverse_s(key):
        return 's' in key and '1/s' in key
    # We combine the above two conditions
    def check_allowed(key):
        return not more_than_one_delta(key) and not s_and_inverse_s(key)

    # We filter out unnecessary combinations.
    final_combinations = filter( check_allowed, combinations )
    return list(final_combinations)

# pylint: disable=too-many-arguments, too-many-locals
def generate_signals(dataset: CurrentVoltageData, emf_function: VoltageFunction,
                     hysteresis_function: VoltageFunction | None, initial_soc: float,
                     sampling_period: float, battery_capacity: float):
    """Generate basic signals, that is, SOC, current, voltage,
    current direction, and hysteresis function using
    the `Signal` class."""
    # 1. Current signal
    current = Signal('i', dataset['current_values'].tolist(), lambda x: x)
    # 3. SOC trajectory possibly dependent on temperature
    soc = Signal('s', [initial_soc],
                 lambda soc, current, delta_t, capacity: soc + delta_t / capacity * current,
                 equation_order=1)
    # Update the trajectory
    for c in current.trajectory[:-1]:
        soc.update_trajectory(c, sampling_period, battery_capacity)
    # 4. Overpotential signal
    # Use EMF trajectory to determine overpotentials.
    emf_trajectory = []
    for s in soc.trajectory[:len(dataset['voltage_values'])]:
        emf_trajectory.append(emf_function(s))
    # Check for nan in the EMF trajectory
    if np.isnan(emf_trajectory).any():
        raise ValueError("Invalid EMF value(s). Possibly out-of-bound SOC input value for "
                         "the EMF function. Note that extrapolation is not allowed.")
    #
    overpotential = Signal('v', (dataset['voltage_values'] - emf_trajectory).tolist(), lambda x: x)
    # 5. Basic current-direction trajectory
    current_direction =  Signal('d', np.sign(np.array(current.trajectory)).tolist(), lambda x: x)
    #
    signals = SignalVector([ soc, current, overpotential, current_direction ])
    #
    if hysteresis_function is not None:
        # 5. Hysteresis-input
        hysteresis_input = Signal('h', [hysteresis_function(s) for s in soc.trajectory],
                                  lambda x: x, equation_order=0)
        signals.add(hysteresis_input)
    #
    return signals

def generate_basis_functions(basis_functions: list[BasisFunction], signals: SignalVector):
    """Generate basis functions using `Signal` class."""
    # Second, we generate basis-function trajectories
    basis_function_signals = []
    for basis_function in basis_functions:
        basic_signal = signals.find(basis_function.variable)
        #
        if basic_signal is None:
            raise ValueError('Symbol not recognized.')
        #
        basis_function_signal = perform_signal_operation(basic_signal, basis_function.operation,
                                                         basis_function.arguments,
                                                         basis_function.function_string)
        basis_function_signals.append( basis_function_signal )
    #
    return SignalVector(basis_function_signals)

def generate_io_trajectories(io_signal_vector: list[Signal], time_delays, model_order):
    """Generate a dictionary of input-output signal trajectories
    including their delayed versions."""
    io_trajectories_dict = {}
    for n_s, signal in enumerate(io_signal_vector):
        for delay in time_delays[n_s]:
            key = signal.symbol + ('(k-' + str(delay) + ')' if delay > 0 else '(k)')
            indices = (model_order - delay, -delay if delay > 0 else None)
            io_trajectories_dict[key] = signal.trajectory[indices[0]:indices[-1]]
    #
    return io_trajectories_dict

def generate_basis_trajectories(basis_functions: list[Signal], time_delays, model_order):
    """Generate a dictionary of basis-function trajectories
    including their delayed versions."""
    p_trajectories_dict = {}
    possible_time_delays = sorted({d for delays in time_delays for d in delays})
    for signal in basis_functions:
        for delay in possible_time_delays:
            key = signal.symbol + ('(k-' + str(delay) + ')' if delay > 0 else '(k)')
            indices = (model_order - delay, -delay if delay > 0 else None)
            p_trajectories_dict[key] = signal.trajectory[indices[0]:indices[-1]]
    #
    return p_trajectories_dict

def generate_signal_trajectories(signals: Tuple[list[Signal], ...], model_order: int,
                                 no_of_initial_values: int):
    """Generate all relevant signal trajectories including input-output,
    basis-function as well as hysteresis basis-function trajectories."""
    #
    io_signals, basis_function_signals, hysteresis_basis_function_signals = signals
    #
    time_delays = [np.arange(1, model_order + 1, 1).tolist(), # voltage
                    np.arange(0, model_order + 1, 1).tolist(), # current
                    [0]] # hysteresis input
    #
    io_trajectories_dict = generate_io_trajectories(io_signals,
                                                    time_delays, no_of_initial_values)
    p_trajectories_dict = generate_basis_trajectories(basis_function_signals,
                                                        time_delays, no_of_initial_values)
    h_trajectories_dict = generate_basis_trajectories(hysteresis_basis_function_signals,
                                                        time_delays, no_of_initial_values)
    #
    return io_trajectories_dict, p_trajectories_dict, h_trajectories_dict
