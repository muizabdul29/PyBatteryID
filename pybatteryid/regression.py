"""Utilities regarding regression."""


from typing import Literal

import numpy as np

from cvxopt import matrix
from sklearn.linear_model import LassoCV, RidgeCV

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

                combined_key = key + "×" + basis_key + delay
                column_labels.append( combined_key )
        else:
            for basis_key in basis_function_keys:
                #
                p_trajectories_list = [ p_trajectories[term + delay] for term in basis_key ]
                phi.append( np.prod([trajectory, *p_trajectories_list], axis=0) )

                combined_key = key + "×" + "×".join([ term + delay for term in basis_key ])
                column_labels.append( combined_key )
    #
    phi = np.array(phi).T
    #
    return phi, np.array(column_labels)


# pylint: disable=too-many-locals
def combine_regression_problems(problems: list[tuple[np.ndarray, ...]],
                                strategy: Literal['concatenate', 'interleave']='interleave'):
    """Combine regression problems A1*x=y1, A2*x=y2, and so on."""
    # possible schemes: append, interleave
    if strategy == 'interleave':
        # Lengths for all regression problems
        problem_sizes = sorted(set(len(problem[1]) for problem in problems))
        # Since the regression problems may not of same size, we
        # split the problems into subproblems such that in each split,
        # the problems are of the same length.
        regression_matrix_parts = []
        output_vector_parts = []
        # We split the problems into subproblems with
        # equal lengths
        for i, problem_size in enumerate(problem_sizes):
            start_idx = problem_sizes[i - 1] if i > 0 else 0
            end_idx = problem_size
            #
            problems_split = [(problem[0][start_idx:end_idx], problem[1][start_idx:end_idx])
                              for problem in problems if len(problem[1][start_idx:end_idx])]
            # Number of problems in the split (referred to as p' in the paper)
            number_of_problems = len(problems_split)
            # Remove all unwanted rows and elements from the regression matrices
            # and the output vectors, respectively.
            regression_matrices = [problem[0][i::number_of_problems]
                                   for i, problem in enumerate(problems_split)]
            output_vectors = [problem[1][i::number_of_problems]
                              for i, problem in enumerate(problems_split)]

            # Initialize final regression matrix and output vector.
            regression_matrix_split = np.empty_like(problems_split[0][0])
            output_vector_split = np.empty_like(problems_split[0][1])
            # Insert rows and elements alternatively (round-robin style) in the regression
            # matrix and output vector, respectively.
            for i, (regression_matrix, output_vector) in enumerate(zip(regression_matrices,
                                                                       output_vectors)):
                if len(regression_matrix) == 0:
                    continue
                regression_matrix_split[i::number_of_problems] = regression_matrix
                output_vector_split[i::number_of_problems] = output_vector
            #
            regression_matrix_parts.append(regression_matrix_split)
            output_vector_parts.append(output_vector_split)
            #
        final_regression_matrix = np.concatenate(regression_matrix_parts)
        final_output_vector = np.concatenate(output_vector_parts)
    elif strategy == 'concatenate':
        regression_matrices = [problem[0] for problem in problems]
        output_vectors = [problem[1] for problem in problems]
        #
        final_regression_matrix = np.concatenate(regression_matrices)
        final_output_vector = np.concatenate(output_vectors)
    else:
        raise ValueError('Invalid strategy to combine regression problems.')
    #
    rich_print(Panel(("Inverse condition number of regression matrix: "
                      f"{1 / np.linalg.cond(final_regression_matrix)}"
                      "\nDimensions of regression matrix: "
                      f"({final_regression_matrix.shape[0]} rows,"
                      f" {final_regression_matrix.shape[1]} columns)")))
    #
    return final_regression_matrix, final_output_vector


def run_optimizer(regression_matrix: np.ndarray, output_vector: np.ndarray, optimizer: str):
    """Run an optimization routine for a regression problem to
    estimate parameters."""
    #
    if optimizer == 'lasso.cvxopt':
        #
        rich_print(Panel(("Performing LASSO using `l1regls.py` from cvxopt.org "
                          "using lambda_1 = 1. See the following link for details:"
                          "\nhttps://cvxopt.org/examples/mlbook/l1regls.html")))
        #
        estimate = l1regls(matrix(regression_matrix), matrix(output_vector))
        estimate = np.array(estimate).flatten()
    elif optimizer == 'lassocv.sklearn':
        #
        rich_print(Panel(("Performing cross-validated LASSO using `sklearn` package.")))
        #
        alphas = np.linspace(1, 5, num=5 ) / (2 * regression_matrix.shape[0])
        model_fit = LassoCV(fit_intercept=False, alphas=alphas, max_iter=1000000000,
                            tol=0.1, verbose=True).fit(regression_matrix, output_vector)
        estimate = np.array(model_fit.coef_).flatten()
        #
        rich_print("Solution found using [bold]lambda_1 = 2 * alpha * n_samples = "
                   f"{2 * model_fit.alpha_ * regression_matrix.shape[0]:.2f}")
    elif optimizer == 'ridgecv.sklearn':
        #
        rich_print(Panel("Performing cross-validated Ridge regression using `sklearn` package."))
        #
        alphas = np.logspace(-3, 1, num=5 )
        model_fit = RidgeCV(fit_intercept=False, cv=10, alphas=alphas).fit(regression_matrix,
                                                                           output_vector)
        estimate = model_fit.coef_
        #
        rich_print(f"Solution found using [bold]lambda_2 = alpha = {model_fit.alpha_}")
    else:
        raise ValueError('Unknown optimization routine.')
    #
    return estimate
