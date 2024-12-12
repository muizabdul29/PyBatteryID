"""
Several utilities for user's convenience.
"""


from dataclasses import asdict, fields

import numpy as np

from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from rich.jupyter import print as rich_print
from rich.table import Table

from .typeddicts import CurrentVoltageData
from .dataclasses import VoltageFunction, Model
from .basisfunctions import generate_signals
from .plotter import plot_custom


def analyze_temperature_soc_space(datasets: list[CurrentVoltageData], battery_capacity: float,
                                  sampling_period: float, emf_function: VoltageFunction,
                                  hysteresis_function: VoltageFunction | None = None):
    """Find the SOC range of given dataset(s) """
    #
    soc_temperature_tuples = []
    soc_values = []
    temperature_values = []
    for dataset in datasets:
        signals = generate_signals(dataset, battery_capacity, sampling_period,
                               emf_function, hysteresis_function)
        soc_signal = signals.find('s').trajectory
        temperature_signal = signals.find('T').trajectory
        #
        soc_temperature_tuples.append((soc_signal, temperature_signal))
        soc_values = np.concatenate((soc_values, soc_signal))
        temperature_values = np.concatenate((temperature_values, temperature_signal))
    #
    points = np.column_stack((soc_values, temperature_values))
    hull = ConvexHull(points)
    #
    simplex_tuples = []
    for simplex in hull.simplices:
        simplex_tuples.append((points[simplex, 0], points[simplex, 1]))
    #
    plot_custom(simplex_tuples + soc_temperature_tuples, xlabel=r'SOC [$\%$]',
                ylabel=r'Temperature [$^\circ$C]', colors=['k'] * len(simplex_tuples),
                figsize=(6, 2), xaxis_reverse=True)


def invert_voltage_function(voltage_function: VoltageFunction, voltage: float,
                            temperature: float | None = None):
    """Determine SOC value corresponding to voltage value."""
    def _function(soc: float):
        return abs(voltage_function(soc, temperature) - voltage)
    #
    return minimize_scalar(_function, bounds=(0, 1)).x


def analyze_dataset(dataset: CurrentVoltageData, battery_capacity: float,
                    sampling_period: float, emf_function: VoltageFunction,
                    hysteresis_function: VoltageFunction | None = None,
                    signals_to_plot: list[str] | None = None):
    """Print basic details of the given dataset such as SOC range,
    cycled capacity, voltage range, etc."""
    #
    signals = generate_signals(dataset, battery_capacity, sampling_period,
                               emf_function, hysteresis_function)
    # Print a table
    table = Table(title="Dataset analysis", show_header=False)
    # Number of samples
    number_of_samples = len(dataset['time_values'])
    table.add_row("Number of samples", f"{number_of_samples}")
    # Time in hours
    experiment_time = dataset['time_values'][-1] - dataset['time_values'][0]
    table.add_row("Experiment time [s], [h]",
                  f"{experiment_time:.3f}, {experiment_time / 3600:.3f}")
    # Cycled capacity
    cycled_capacity = trapezoid(dataset['current_values'])
    table.add_row("Extracted charge [C], [Ah]",
                  f"{-cycled_capacity:.3f}, {-cycled_capacity / 3600:.5f}")
    # Initial voltage value
    initial_voltage = dataset['voltage_values'][0]
    table.add_row("Initial voltage [V]", f"{initial_voltage:.5f}")
    # Voltage range
    voltage = dataset['voltage_values']
    table.add_row("Voltage range [V]", f"({min(voltage):.5f}, {max(voltage):.5f})")
    # SOC range
    soc = signals.find('s').trajectory
    table.add_row("SOC range [%]", f"({min(soc) * 100:.2f}, {max(soc) * 100:.2f})")
    # Temperature range
    if 'temperature_values' in dataset:
        temperature = dataset['temperature_values']
        table.add_row("Temperature range [°C]", f"({min(temperature):.2f}, {max(temperature):.2f})")
    #
    rich_print(table, markup=False)
    #
    if signals_to_plot:
        for symbol in signals_to_plot:
            signal_trajectory = signals.find(symbol).trajectory
            plot_custom([(dataset['time_values'], signal_trajectory)], figsize=(6, 2),
                        xlabel='Time', ylabel=symbol, colors=['darkblue'])


def save_model_to_file(model: Model, path_to_directory: str, description: str):
    """Save a model to file."""
    file_name = f'{description}_n,l={model.model_order},{model.nonlinearity_order}'
    file_path = f'{path_to_directory}/{file_name}'

    np.save(file_path, asdict(model)) # type: ignore


def load_model_from_file(file_path: str):
    """Load a model from file."""
    model_as_dict = np.load(f'{file_path}', allow_pickle=True).item()
    #
    model = _dataclass_from_dict(Model, model_as_dict)
    if not isinstance(model, Model):
        raise ValueError('Cannot load model.')
    return model


def print_model_details(model: Model):
    """Print model details."""
    # We print the details as a table.
    table = Table(title=(f"Model order = {model.model_order}; "
                         f"Nonlinearity order = {model.nonlinearity_order}"))
    #
    table.add_column("Model Term")
    table.add_column("Estimated Parameter")
    #
    for estimate, term in zip(model.model_estimate, model.model_terms):
        table.add_row(str(term), str(estimate))
    #
    rich_print(table, markup=False)


def _dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in fields(klass)}
        return klass(**{f:_dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except: # pylint: disable=bare-except
        return d # Not a dataclass field
