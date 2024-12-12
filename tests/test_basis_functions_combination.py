"""
Testing related to combination of basis functions from
user-provided strings.
"""

from pybatteryid.basisfunctions import combine_symbols

def test_s_combination():
    """Test combinations involving s variables."""
    symbols = ['s', '1/s']
    combinations = combine_symbols(symbols, 3)

    assert combinations == [('s',), ('1/s',), ('s', 's'), ('1/s', '1/s'),
                            ('s', 's', 's'), ('1/s', '1/s', '1/s')]

def test_d_combination():
    """Test combinations involving d variables."""
    symbols = ['d[0.01,0.99]', 'd[0.001,0.999]']
    combinations = combine_symbols(symbols, 10)

    assert combinations == [('d[0.01,0.99]',), ('d[0.001,0.999]',)]

def test_s_and_d_combination():
    """Test combinations involving s and d variables."""
    symbols = ['s', '1/s', 'd[0.001,0.999]']
    combinations = combine_symbols(symbols, 2)

    assert combinations == [('s',), ('1/s',), ('d[0.001,0.999]',), ('s', 's'),
                            ('s', 'd[0.001,0.999]'), ('1/s', '1/s'), ('1/s', 'd[0.001,0.999]')]
