from unittest import TestCase
import numpy as np
from local_circuit.circuit_config import CircuitConfig

from numpy.testing import assert_array_almost_equal

from local_circuit.stp import update_stp_weights


class CircuitTest(TestCase):
    def test_update_stp_weights_empty(self):
        stp_weights = np.empty((0, 0))
        spike_steps = np.empty(0)
        config = CircuitConfig(N=0)
        update_stp_weights(stp_weights, spike_steps, config)

    def test_update_stp_weights_simple(self):
        stp_weights = np.array(
            [[0.0, 0.2, 0.3], [0.4, 0.0, 0.6], [0.7, 0.8, 0.0]])
        spike_steps = np.array([1, -1, 0])
        config = CircuitConfig(N=3, stp_decay_factor=0.5)
        update_stp_weights(stp_weights, spike_steps, config)

        expected_stp_weights = np.array(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 0.3], [0.0, 0.0, 0.0]])

        assert_array_almost_equal(stp_weights, expected_stp_weights)

    def test_update_stp_weights_complex(self):
        stp_weights = np.full((5, 5), 0.5)
        config = CircuitConfig(N=5, stp_decay_factor=0.5)

        spike_steps = np.array([1, 0, 1, -1, 2])

        update_stp_weights(stp_weights, spike_steps, config)

        expected_stp_weights = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0.25, 0.25, 0.25, 0.25, 0.25],
            [1, 1, 1, 0, 0]
        ])

        assert_array_almost_equal(stp_weights, expected_stp_weights)
