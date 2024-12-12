"""
Testing related to generation of basis functions from
user-provided strings.
"""

from pybatteryid.basisfunctions import Signal, generate_io_trajectories, \
    generate_basis_trajectories

def test_io_trajectories():
    """Test IO trajectories."""
    voltage = Signal('v', [1, 2, 3, 4, 5], lambda x: x)
    current = Signal('i', [10, 11, 12, 13, 14], lambda x: x)

    time_delays = [ [1, 2], [0, 1, 2] ]

    trajectories_dict = generate_io_trajectories([voltage, current], time_delays, 2)

    assert trajectories_dict['i(k)'] == [12, 13, 14]
    assert trajectories_dict['v(k-2)'] == [1, 2, 3]
    assert trajectories_dict['i(k-1)'] == [11, 12, 13]

def test_basis_trajectories():
    """Test basis trajectories."""
    soc = Signal('s', [100, 200, 300, 400, 500], lambda x: x)

    time_delays = [ [1, 2], [0, 1, 2] ]

    trajectories_dict = generate_basis_trajectories([soc], time_delays, 2)

    assert trajectories_dict['s(k)'] == [3, 4, 5]
    assert trajectories_dict['s(k-1)'] == [2, 3, 4]
    assert trajectories_dict['s(k-2)'] == [1, 2, 3]
