from unittest import TestCase
import numpy as np
from local_circuit.circuit_config import CircuitConfig

from numpy.testing import assert_array_almost_equal

from local_circuit.ltp import update_ltp_weights


class CircuitTest(TestCase):
    def test_update_ltp_weights_empty(self):
        ltp_weights = np.empty((0, 0))
        spike_steps = np.empty(0)
        config = CircuitConfig(N=0)
        update_ltp_weights(ltp_weights, spike_steps, config)

    def test_update_ltp_weights_simple(self):
        ltp_weights = np.array(
            [[0.0, 0.2, 0.3], [0.4, 0.0, 0.6], [0.7, 0.8, 0.0]])
        spike_steps = np.array([1, -1, 0])
        config = CircuitConfig(N=3, ltp_step_up=0.01, ltp_step_down=0.02)
        update_ltp_weights(ltp_weights, spike_steps, config)

        expected_ltp_weights = np.array(
            [[0.0, 0.2, 0.31], [0.38, 0.0, 0.58], [0.68, 0.8, 0.0]])

        assert_array_almost_equal(ltp_weights, expected_ltp_weights)

    def test_update_ltp_weights_clipping(self):
        ltp_weights = np.array([[0.0, 0.4, 0.7],
                                [0.2, 0.0, 0.8],
                                [0.3, 0.6, 0.0]])
        spike_steps = np.array([1, -1, 0])
        config = CircuitConfig(N=3, ltp_step_up=0.4, ltp_step_down=0.25)
        update_ltp_weights(ltp_weights, spike_steps, config)

        expected_ltp_weights = np.array(
            [[0.0, 0.4, 1.0], [0.0, 0.0, 0.55], [0.05, 0.6, 0.0]])

        assert_array_almost_equal(ltp_weights, expected_ltp_weights)

    def test_update_ltp_weights_complex(self):
        ltp_weights = np.full((5, 5), 0.5)
        config = CircuitConfig(N=5, ltp_step_up=0.1, ltp_step_down=0.2)

        spike_steps = np.array([1, 0, 1, -1, 2])

        update_ltp_weights(ltp_weights, spike_steps, config)
        expected_ltp_weights = np.full((5, 5), 0.5)

        # effects from neuron 0
        expected_ltp_weights[1, 0] = 0.3
        expected_ltp_weights[2, 0] = 0.3
        expected_ltp_weights[3, 0] = 0.3
        expected_ltp_weights[4, 0] = 0.6

        # effects from neuron 1
        expected_ltp_weights[0, 1] = 0.6
        expected_ltp_weights[2, 1] = 0.6
        expected_ltp_weights[3, 1] = 0.3
        expected_ltp_weights[4, 1] = 0.6

        # effects from neuron 2
        expected_ltp_weights[0, 2] = 0.3
        expected_ltp_weights[1, 2] = 0.3
        expected_ltp_weights[3, 2] = 0.3
        expected_ltp_weights[4, 2] = 0.6

        # effects from neuron 4
        expected_ltp_weights[0, 4] = 0.3
        expected_ltp_weights[1, 4] = 0.3
        expected_ltp_weights[2, 4] = 0.3
        expected_ltp_weights[3, 4] = 0.3

        assert_array_almost_equal(ltp_weights, expected_ltp_weights)
