"""
Testing related to extraction of basis functions from
user-provided strings.
"""

from pybatteryid.basisfunctions import Operation, extract_basis_functions

def test_no_operation_basis_function_extraction():
    """Test if the basis functions with no operation
    are extracted correctly"""

    strings = ['s']
    basis_functions = extract_basis_functions(strings)

    assert basis_functions[0].variable == 's'
    assert basis_functions[0].operation == Operation.NONE
    assert len(basis_functions[0].arguments) == 0
    assert basis_functions[0].function_string == 's'

def test_inverse_operation_basis_function_extraction():
    """Test if the basis functions with inverse operation
    are extracted correctly"""

    strings = ['1/s']
    basis_functions = extract_basis_functions(strings)

    assert basis_functions[0].variable == 's'
    assert basis_functions[0].operation == Operation.INVERSE
    assert len(basis_functions[0].arguments) == 0
    assert basis_functions[0].function_string == '1/s'

def test_logarithm_operation_basis_function_extraction():
    """Test if the basis functions with logarithm operation
    are extracted correctly"""

    strings = ['log[T]']
    basis_functions = extract_basis_functions(strings)

    assert basis_functions[0].variable == 'T'
    assert basis_functions[0].operation == Operation.LOGARITHM
    assert len(basis_functions[0].arguments) == 0
    assert basis_functions[0].function_string == 'log[T]'

def test_exp_sqrt_abs_operation_basis_function_extraction():
    """Test if the basis functions with exponential/square root/absolute
    operations are extracted correctly"""

    strings = ['exp[0.05*sqrt[|i|]]', 'exp[-2.759*sqrt[0.5*s-140]]']
    basis_functions = extract_basis_functions(strings)

    assert basis_functions[0].variable == 'i'
    assert basis_functions[0].operation == Operation.EXP_SQRT_ABS
    assert basis_functions[0].arguments == ['0.05', '', '', '|', '|', 'sqrt']
    assert basis_functions[0].function_string == 'exp[0.05*sqrt[|i|]]'

    assert basis_functions[1].variable == 's'
    assert basis_functions[1].operation == Operation.EXP_SQRT_ABS
    assert basis_functions[1].arguments == ['-2.759', '0.5', '-140', '', '', 'sqrt']
    assert basis_functions[1].function_string == 'exp[-2.759*sqrt[0.5*s-140]]'

def test_exp_power_abs_operation_basis_function_extraction():
    """Test if the basis functions with exponential/power/absolute
    operations are extracted correctly"""

    strings = ['exp[[0.00366*T+1]^-1]', 'exp[-2.759*[|-1.5*i+1890|]^+2.89]']
    basis_functions = extract_basis_functions(strings)

    assert basis_functions[0].variable == 'T'
    assert basis_functions[0].operation == Operation.EXP_POWER_ABS
    assert basis_functions[0].arguments == ['', '0.00366', '+1', '', '', '-1']
    assert basis_functions[0].function_string == 'exp[[0.00366*T+1]^-1]'

    assert basis_functions[1].variable == 'i'
    assert basis_functions[1].operation == Operation.EXP_POWER_ABS
    assert basis_functions[1].arguments == ['-2.759', '-1.5', '+1890', '|', '|', '+2.89']
    assert basis_functions[1].function_string == 'exp[-2.759*[|-1.5*i+1890|]^+2.89]'
