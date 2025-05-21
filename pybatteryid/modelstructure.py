"""
Contains the class `ModelStructure`.
"""


from .voltage import load_voltage_model, VoltageFunction
from .basisfunctions import extract_basis_functions
from .dataclasses import BasisFunction
from .typeddicts import VoltageSocData


class ModelStructure:
    """This class allows battery model identification and simulation employing
    the proposed model structure using the input-output (IO) representation in
    the linear parameter-varying (LPV) framework.

    Attributes
    ----------
    battery_capacity : float
        Capacity of the battery being modelled.
    sampling_period: float
        Model sampling time.
    """

    battery_capacity: float
    sampling_period: float

    emf_function: VoltageFunction
    hysteresis_function: VoltageFunction | None

    basis_functions: list[BasisFunction]
    hysteresis_basis_functions: list[BasisFunction]


    def __init__(self, battery_capacity: float, sampling_period: float):
        self.battery_capacity = battery_capacity
        self.sampling_period = sampling_period

        self.hysteresis_function = None

        self.basis_functions = []
        self.hysteresis_basis_functions = []


    def add_emf_function(self, voltage_soc_data: VoltageSocData):
        """Add EMF function used to decompose battery voltage into
        overpotentials and vice versa."""
        #
        self.emf_function = load_voltage_model(voltage_soc_data)


    def add_hysteresis_function(self, voltage_soc_data: VoltageSocData):
        """Add hysteresis function to be used as second model input."""
        #
        self.hysteresis_function = load_voltage_model(voltage_soc_data)


    def add_basis_functions(self, basis_function_strings: list[str],
                            hysteresis_basis_function_strings: list[str] | None = None):
        """Add basis functions by extracting signal variable, the desired
        operation and its arguments."""
        self.basis_functions = extract_basis_functions(basis_function_strings)
        if hysteresis_basis_function_strings is not None:
            hbfs = hysteresis_basis_function_strings
            self.hysteresis_basis_functions = extract_basis_functions(hbfs)
