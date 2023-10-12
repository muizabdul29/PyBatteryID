"""
Contains class `InputDesign`
"""

import numpy as np

# pylint: disable=too-many-instance-attributes, too-few-public-methods
class InputDesign:
    """This class allows generation of current profile employed as
    identification input using the proposed design in [X]. Note
    that the model output is the battery voltage, and the model
    application is assumed to involve constant temperature."""

    _mean_pulse_period: float
    _std_pulse_period: float

    _mean_rest_period: float
    _std_rest_period: float

    _initial_amplitude: float

    _amplitude_plus: float
    _amplitude_minus: float

    _std_random_amplitude: float

    _dc_offset: float

    _no_of_pulses: int

    _upper_limit: float | None
    _lower_limit: float | None

    # pylint: disable=too-many-arguments
    def __init__(self,
                 mean_pulse_period: float,
                 std_pulse_period: float,
                 mean_rest_period: float,
                 std_rest_period: float,
                 initial_amplitude: float,
                 amplitude_plus: float,
                 amplitude_minus: float,
                 std_random_amplitude: float,
                 dc_offset: float,
                 no_of_pulses: int,
                 upper_safety_limit: float | None = None,
                 lower_safety_limit: float | None = None) -> None:
        """Load required signal parameters.

        Parameters:
        ----------
            mean_pulse_period : float
                `μ_tau` - Mean pulse period.
            std_pulse_period : float
                `σ_tau` - Standard deviation in the pulse period.
            mean_rest_period : float
                `μ_tau0` - Mean rest period.
            std_rest_period : float
                `σ_tau0` - Standard deviation in the rest period.
            initial_amplitude : float
                `alpha0` - Initial amplitude of the pulse train.
            amplitude_plus : float
                `alpha+` - Upper limit on the final amplitude `alpha`.
            amplitude_minus : float
                `alpha-` - Lower limit on the final amplitude `alpha`.
            std_random_amplitude : float
                `σ_xi` - Standard deviation in the randomness added to
                the signal.
            dc_offset : float
                `nu` - DC offset added to the pulse train.
            no_of_pulses : int
                Number of pulses in the signal.
            upper_safety_limit : float | None
                Upper limit on the identification signal for safety reasons.
            lower_safety_limit : float | None
                Lower limit on the identification signal for safety reasons.
        """
        self._mean_pulse_period = mean_pulse_period
        self._std_pulse_period = std_pulse_period

        self._mean_rest_period = mean_rest_period
        self._std_rest_period = std_rest_period

        self._initial_amplitude = initial_amplitude

        self._amplitude_plus = amplitude_plus
        self._amplitude_minus = amplitude_minus

        self._std_random_amplitude = std_random_amplitude

        self._dc_offset = dc_offset

        self._no_of_pulses = no_of_pulses

        self._upper_limit = upper_safety_limit
        self._lower_limit = lower_safety_limit

    # pylint: disable=too-many-locals
    def generate_signal_profile(self):
        """Generate signal profile used as identification input."""
        # First, we generate periods for each pulse.
        pulse_periods = np.round(np.abs(np.random.normal(self._mean_pulse_period,
                                                         self._std_pulse_period,
                                                         self._no_of_pulses)))
        # Time instants (with sampling period = 1 sec)
        time = np.arange(np.sum(pulse_periods))
        # Starting time instants of pulses
        pulse_start_times = np.cumsum(pulse_periods[:-1])
        pulse_start_times = np.insert(pulse_start_times, 0, 0)

        # Generate rectangular shape by setting 1s' between
        # starting times and half of the corresponding period.
        # Later, it is shifted downward by 0.5, and then multiplied
        # with 2 to scale the amplitudes to unity. The minus ensures
        # the discharge pulse phase at the beginning.
        rectangular_shape = (time >= pulse_start_times[:, np.newaxis]) & \
            (time < pulse_start_times[:, np.newaxis] + 0.5 * pulse_periods[:, np.newaxis])
        pulse_train = -2 * (np.sum(rectangular_shape, axis=0 ) - 0.5)

        # Split pulse train into list of phases
        phase_start_times = np.ceil(np.cumsum(np.repeat(pulse_periods, 2) / 2)).astype(int)
        phases = np.split(pulse_train, phase_start_times[:-1])

        # Add rest period following each phase
        rest_periods = np.rint(np.abs(np.random.normal(self._mean_rest_period,
                                                       self._std_rest_period,
                                                       len(phases)))).astype(int)
        # Define the final amplitude of the signal
        # NOTE: This generates random number per pulse, and not per phase,
        # that is, we generate unique random numbers half the number of
        # phases, and then repeat them.
        final_amplitudes = np.repeat(np.random.uniform(self._amplitude_minus,
                                                       self._amplitude_plus,
                                                       int(len(phases) / 2)), 2)
        # We finally construct the identification signal
        identification_signal = []
        for i, phase in enumerate(phases):
            # First, we define a sequence of zeros to add rest period
            # following each pulse.
            rest_signal = np.zeros(rest_periods[i])
            # Now, we add zero-mean random signal for high-frequency
            # excitations.
            zero_mean_random_signal = np.random.normal(0, self._std_random_amplitude, len(phase))

            finalized_phase = final_amplitudes[i] * (
                self._initial_amplitude * phase + self._dc_offset + zero_mean_random_signal
            )

            # Applying safety limits if provided
            if self._upper_limit is not None:
                finalized_phase[finalized_phase > self._upper_limit] = self._upper_limit
            if self._lower_limit is not None:
                finalized_phase[finalized_phase < self._lower_limit] = self._lower_limit

            identification_signal = np.concatenate((identification_signal,
                                                    finalized_phase,
                                                    rest_signal))
        #
        return identification_signal
