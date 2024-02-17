from unittest import TestCase
import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from local_circuit.circuit import Circuit
from local_circuit.circuit_config import CircuitConfig


class CircuitTest(TestCase):
    def test_get_effective_weights(self):
        circuit = Circuit(
            CircuitConfig(N=2, weight_scale_factor=0.1, stp_skew_factor=0.7)
        )
        circuit.lt_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        circuit.st_weights = np.array([[0.9, 0.8], [0.7, 0.6]])
        weights = circuit._get_effective_weights()
        expected_weights = np.array([[0.066, 0.062], [0.058, 0.054]])
        assert_array_almost_equal(weights, expected_weights)

    def test_window_rates(self):
        circuit = Circuit(
            CircuitConfig(N=2, weight_scale_factor=0.0, window_rate_decay_factor=0.9)
        )

        assert_array_almost_equal(circuit.window_rates, np.array([0.0, 0.0]))
        circuit.process_cycle(np.array([]))
        assert_array_almost_equal(circuit.window_rates, np.array([0.0, 0.0]))
        circuit.process_cycle(np.array([1]))
        assert_array_almost_equal(circuit.window_rates, np.array([0.0, 0.1]))
        circuit.process_cycle(np.array([]))
        assert_array_almost_equal(circuit.window_rates, np.array([0.0, 0.09]))
        circuit.process_cycle(np.array([0, 1]))
        assert_array_almost_equal(
            circuit.window_rates, np.array([0.1, 0.09 * 0.9 + 0.1])
        )

    def test_window_rates_convergence(self):
        circuit = Circuit(
            CircuitConfig(N=1, weight_scale_factor=0.0, window_rate_decay_factor=0.99)
        )

        for _ in range(100):
            circuit.process_cycle(np.array([0]))
            for _ in range(3):
                circuit.process_cycle(np.array([]))

        assert_allclose(circuit.window_rates[0], 0.25, 0.1)

    def test_homeostasis(self):
        circuit = Circuit(
            CircuitConfig(
                N=1,
                homeostasis_target=0.2,
                homeostasis_scale_factor=100,
                window_rate_decay_factor=0.9,
            )
        )

        assert_almost_equal(circuit._compute_homeostasis_offsets()[0], 0.2 * 100)
        assert_almost_equal(circuit.window_rates[0], 0.0)
        circuit.process_cycle(np.array([]))
        assert_almost_equal(circuit._compute_homeostasis_offsets()[0], 0.1 * 100)
        assert_almost_equal(circuit.window_rates[0], 0.1)
        circuit.process_cycle(np.array([]))
        assert_almost_equal(circuit._compute_homeostasis_offsets()[0], 0.01 * 100)
        assert_almost_equal(circuit.window_rates[0], 0.19)
        circuit.process_cycle(np.array([]))
        assert_almost_equal(circuit._compute_homeostasis_offsets()[0], -0.071 * 100)
        assert_almost_equal(circuit.window_rates[0], 0.271)
        circuit.process_cycle(np.array([]))
        assert_almost_equal(circuit._compute_homeostasis_offsets()[0], -0.0439 * 100)
        assert_almost_equal(circuit.window_rates[0], 0.2439)
        circuit.process_cycle(np.array([]))
        assert_almost_equal(circuit._compute_homeostasis_offsets()[0], -0.01951 * 100)
        assert_almost_equal(circuit.window_rates[0], 0.21951)

    def test_tf_spike(self):
        circuit = Circuit(CircuitConfig(N=2, weight_scale_factor=0.0, ltp_step_up=0.1))
        circuit.lt_weights = np.zeros((2, 2))
        circuit.st_weights = np.zeros((2, 2))

        circuit.process_cycle(np.array([1]))

        assert_array_almost_equal(circuit.lt_weights, np.zeros((2, 2)))
        assert_array_almost_equal(circuit.st_weights, np.zeros((2, 2)))

        out_frame = circuit.process_cycle(np.array([1]), tf_spike=np.array([0]))

        expected_lt_weights = np.array([[0, 0.1], [0, 0]])
        expected_st_weights = np.array([[0, 1], [0, 0]])

        assert_array_almost_equal(circuit.lt_weights, expected_lt_weights)
        assert_array_almost_equal(circuit.st_weights, expected_st_weights)
        assert_array_equal(out_frame, np.array([1]))

    def test_tf_no_spike(self):
        circuit = Circuit(
            CircuitConfig(
                N=2,
                weight_scale_factor=2.0,
                ltp_step_up=0.1,
                ltp_step_down=0.3,
                stp_skew_factor=0.0,
            )
        )
        circuit.lt_weights = np.array([[0.0, 0.5], [0.0, 0.0]])
        circuit.st_weights = np.zeros((2, 2))

        out_frame = circuit.process_cycle(np.array([1]), tf_no_spike=np.array([0]))

        assert_array_almost_equal(circuit.lt_weights, np.array([[0, 0.2], [0, 0]]))
        assert_array_almost_equal(circuit.st_weights, np.array([[0, 0], [0, 0]]))
        assert_array_equal(out_frame, np.array([0, 1]))

    def test_integration(self):
        cc = CircuitConfig(
            N=10, ltp_step_up=1.0, ltp_step_down=1.0, stp_skew_factor=0.0
        )
        circuit = Circuit(cc)
        circuit.weight_mask = np.zeros_like(circuit.weight_mask)
        circuit.weight_mask[-1, :] = 1.0

        in_frame = np.array([2])
        tf_spike = np.array([9])
        circuit.process_cycle(in_frame, tf_spike)

        in_frame = np.array([0, 1, 3, 4, 5, 6, 7, 8])
        tf_no_spike = np.array([9])
        circuit.process_cycle(in_frame, tf_no_spike=tf_no_spike)

        out_frame = circuit.process_cycle(np.array([2]))

        assert_array_equal(out_frame, np.array([2, 9]))
