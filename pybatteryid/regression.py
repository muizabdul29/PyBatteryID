"""abc"""

import numpy as np

from cvxopt import matrix
from sklearn.linear_model import RidgeCV

from rich.panel import Panel
from rich.jupyter import print as rich_print

from .l1regls import l1regls

def setup_regression(io_trajectories: dict, p_trajectories: dict, h_trajectories: dict,
                     basis_function_keys: list[list[str]],
                     hysteresis_basis_function_keys: list[str]):
    """Setup regression problem by constructing a regression matrix."""
    phi = []
    column_labels = []
    #
    for key, trajectory in io_trajectories.items():
        # extract delay from key
        symbol = key.split("(")[0]
        delay = "(" + key.split("(")[-1]
        #
        phi.append( trajectory )
        column_labels.append(key)

        if symbol == 'h':
            for basis_key in hysteresis_basis_function_keys:
                #
                h_trajectory = h_trajectories[basis_key + delay]
                phi.append( np.prod([trajectory, h_trajectory], axis=0) )

                combined_key = key + "*" + basis_key + delay
                column_labels.append( combined_key )
        else:
            for basis_key in basis_function_keys:
                #
                p_trajectories_list = [ p_trajectories[term + delay] for term in basis_key ]
                phi.append( np.prod([trajectory, *p_trajectories_list], axis=0) )

                combined_key = key + "*" + "*".join([ term + delay for term in basis_key ])
                column_labels.append( combined_key )
    #
    return np.array(phi).T, np.array(column_labels)

def run_optimizer(regression_matrix: np.ndarray, output_vector: np.ndarray, optimizer: str):
    """Run an optimization routine for a regression problem to
    estimate parameters."""
    #
    if optimizer == 'lasso':
        #
        rich_print(Panel(("Performing LASSO using `l1regls.py` from cvxopt.org. "
                          "See the following link for details:"
                          "\nhttps://cvxopt.org/examples/mlbook/l1regls.html")))
        #
        estimate = l1regls(matrix(regression_matrix), matrix(output_vector))
        estimate = np.array(estimate).flatten()
    elif optimizer == 'ridge':
        #
        rich_print(Panel("Performing cross-validated Ridge regression using `sklearn` package."))
        #
        alphas = np.logspace(-3, 1, num=5 )
        model_fit = RidgeCV(fit_intercept=False, cv=10, alphas=alphas).fit(regression_matrix,
                                                                           output_vector)
        estimate = model_fit.coef_
        #
        rich_print(f"Solution found using [bold]alpha = {model_fit.alpha_}")
    else:
        raise ValueError('Unknown optimization routine.')
    #
    return estimate
