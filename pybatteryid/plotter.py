"""
A collection of utilities for battery plots.
"""

from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# pylint: disable=too-many-locals,too-many-arguments
def plot_custom(x_y_data_tuples: list[Tuple], figsize: Tuple[int, int]=(10, 3),
                xlabel: str='x', ylabel: str='y', legends: list[str]|None=None,
                linestyles: list[str]|None=None, xaxis_reverse: bool=False,
                ylims: Tuple[int, int]|None=None, xlims: Tuple[int, int]|None=None,
                title: str|None=None, linewidth: float|None=None,
                colors: list[str]|None=None):
    """Plot utility with various options"""
    #
    plt.figure(figsize=figsize)
    axis: Axes = plt.gca()
    #
    for i, (x_data, y_data) in enumerate(x_y_data_tuples):

        x_data_copy = list(x_data)
        y_data_copy = list(y_data)

        if colors is not None and len(colors) > i:
            axis.plot(x_data_copy, y_data_copy,
                      linestyles[i] if linestyles is not None and len(linestyles) > i else '',
                      color=colors[i] if colors is not None and len(colors) > i else None,
                      lw=linewidth if linewidth is not None else 0.7)
        else:
            axis.plot(x_data_copy, y_data_copy,
                      linestyles[i] if linestyles is not None and len(linestyles) > i else '',
                      lw=linewidth if linewidth is not None else 0.7)
    #
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    #
    axis.grid(linestyle='--', lw=0.5)
    #
    if ylims is not None:
        axis.set_ylim(*ylims)
    #
    if xlims is not None:
        axis.set_xlim(*xlims)
    #
    if legends is not None:
        axis.legend(legends)
    #
    if title is not None:
        axis.set_title(title)
    #
    if xaxis_reverse:
        axis.invert_xaxis()
    #
    plt.show()

def plot_time_vs_voltage(time_voltage_data_tuples: list[Tuple], figsize: Tuple[int, int]=(6, 2),
                         legends: list[str]|None=None, linestyles: list[str]|None=None,
                         linewidth: float|None=None, units: Tuple[str, str]=('sec', 'V'),
                         xlims: Tuple[int, int]|None=None, ylims: Tuple[int, int]|None=None,
                         colors: list[str]|None=None):
    """A shortcut for time vs. voltage plots."""
    #
    plot_custom(time_voltage_data_tuples, figsize, xlabel=f'Time [{units[0]}]',
        ylabel=f'Voltage [{units[1]}]', legends=legends, linestyles=linestyles,
        linewidth=linewidth, xlims=xlims, ylims=ylims, colors=colors)

def plot_time_vs_current(time_current_data_tuples: list[Tuple], figsize: Tuple[int, int]=(6, 2),
                         legends: list[str]|None=None, linestyles: list[str]|None=None,
                         linewidth: float|None=None, units: Tuple[str, str]=('sec', 'A'),
                         xlims: Tuple[int, int]|None=None, ylims: Tuple[int, int]|None=None,
                         colors: list[str]|None=None):
    """A shortcut for time vs. current plots."""
    #
    plot_custom(time_current_data_tuples, figsize, xlabel=f'Time [{units[0]}]',
                ylabel=f'Current [{units[1]}]', legends=legends, linestyles=linestyles,
                ylims=ylims, xlims=xlims, linewidth=linewidth, colors=colors)
