"""
Testing related to formation of regression matrix.
"""

import numpy as np

from pybatteryid.identification import setup_regression_problems
from pybatteryid.modelstructure import ModelStructure


def test_regression_problem_setup():
    """Test formation of regression problem."""

    example_dataset = {'initial_soc': 0.1,
                       'current_values': np.array([1, 2, 3, -2, -1]),
                       'voltage_values': np.array([1, 4, 9, -4, -1])}

    model_structure = ModelStructure(10, 1)
    model_structure.add_emf_function({'soc_values': np.array([0, 1]),
                                      'voltage_values': np.array([1, 1])})
    model_structure.add_basis_functions(['d[0,0]', 'log[s]'])
    #
    r_problems, r_labels = setup_regression_problems(datasets=[example_dataset],
                                                     model_structure=model_structure,
                                                     model_order=1, nonlinearity_order=1)
    r_matrix, r_output_vector = r_problems[0]
    # pylint: disable=C0103
    A = {'d[0,0](k-1)': np.sign(example_dataset['current_values'][:-1]),
         'd[0,0](k)': np.sign(example_dataset['current_values'][1:]),
         'log[s](k-1)': np.log([ 0.1, 0.2, 0.4, 0.7 ]),
         'log[s](k)': np.log([ 0.2, 0.4, 0.7, 0.5 ])}
    op_values = example_dataset['voltage_values'] - 1
    expected_r_matrix = np.column_stack([op_values[:-1],
                                         op_values[:-1] * A['d[0,0](k-1)'],
                                         op_values[:-1] * A['log[s](k-1)'],
                                         example_dataset['current_values'][1:],
                                         example_dataset['current_values'][1:] * A['d[0,0](k)'],
                                         example_dataset['current_values'][1:] * A['log[s](k)'],
                                         example_dataset['current_values'][:-1],
                                         example_dataset['current_values'][:-1] * A['d[0,0](k-1)'],
                                         example_dataset['current_values'][:-1] * A['log[s](k-1)']])
    expected_r_output_vector = op_values[1:]
    expected_r_labels = ['v(k-1)', 'v(k-1)×d[0,0](k-1)', 'v(k-1)×log[s](k-1)',
                         'i(k)', 'i(k)×d[0,0](k)', 'i(k)×log[s](k)',
                         'i(k-1)', 'i(k-1)×d[0,0](k-1)', 'i(k-1)×log[s](k-1)']

    assert np.allclose(r_output_vector, expected_r_output_vector)
    assert np.allclose(r_matrix, expected_r_matrix)
    assert np.array_equal(r_labels, expected_r_labels)
