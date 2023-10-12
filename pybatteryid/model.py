"""
A collection of utilities related to an identified model.
"""

import numpy as np

from rich.jupyter import print as rich_print
from rich.table import Table

from .dataclasses import Model

def save_to_file(model: Model, path_to_directory: str, description: str):
    """Save a model to file."""
    file_name = f'{description}_n,l={model.model_order},{model.nonlinearity_order}'
    file_path = f'{path_to_directory}/{file_name}'

    np.save(file_path, model) # type: ignore

def load_from_file(file_path: str):
    """Load a model from file."""
    model = np.load(f'{file_path}', allow_pickle=True).item()
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
