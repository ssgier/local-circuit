from unittest import TestCase, result
from local_circuit.circuit import process_frame
from local_circuit.circuit_config import CircuitConfig, CompartmentConfig
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestProcessFrame(TestCase):
    def test_empty_frame(self):
        circuit_config = CircuitConfig(
            N=100,
            compartments=[
                CompartmentConfig(neuron_indexes=slice(100), max_spikes_per_cycle=10)
            ],
        )

        in_frame = np.array([])
        effective_weights = np.ones((circuit_config.N, circuit_config.N))
        membrane_voltage_offsets = np.zeros(circuit_config.N)

        result = process_frame(
            circuit_config,
            in_frame,
            effective_weights,
            membrane_voltage_offsets,
            diagnostics=True,
        )

        assert_array_equal(result.out_frame, np.array([]))
        assert_array_equal(result.spike_steps, np.full(100, -1))
        assert result.step_membrane_voltages is not None
        assert_array_almost_equal(result.step_membrane_voltages, np.zeros((1, 100)))

    def test_voltage_offsets(self):
        circuit_config = CircuitConfig(
            N=100,
            compartments=[
                CompartmentConfig(neuron_indexes=slice(100), max_spikes_per_cycle=1)
            ],
        )

        in_frame = np.array([])
        effective_weights = np.zeros((circuit_config.N, circuit_config.N))
        membrane_voltage_offsets = np.zeros(circuit_config.N)
        membrane_voltage_offsets[9] = 1.0
        membrane_voltage_offsets[10] = 1.1

        result = process_frame(
            circuit_config,
            in_frame,
            effective_weights,
            membrane_voltage_offsets,
            diagnostics=True,
        )

        expected_spike_steps = np.full(100, -1)
        expected_spike_steps[10] = 0

        expected_step_membrane_voltages = np.zeros((2, 100))
        expected_step_membrane_voltages[0] = membrane_voltage_offsets
        expected_step_membrane_voltages[1, 9] = 1.0

        assert_array_equal(result.out_frame, np.array([10]))
        assert_array_equal(result.spike_steps, expected_spike_steps)
        assert result.step_membrane_voltages is not None
        assert_array_almost_equal(
            result.step_membrane_voltages,
            expected_step_membrane_voltages,
        )

    def test_strong_input(self):
        circuit_config = CircuitConfig(
            N=100,
            compartments=[
                CompartmentConfig(neuron_indexes=slice(100), max_spikes_per_cycle=4)
            ],
        )

        in_frame = np.arange(100)
        effective_weights = np.zeros((circuit_config.N, circuit_config.N))
        membrane_voltage_offsets = np.zeros(circuit_config.N)

        membrane_voltage_offsets[2] = 0.1
        membrane_voltage_offsets[4] = 0.1
        membrane_voltage_offsets[7] = 0.1
        membrane_voltage_offsets[9] = 0.1

        result = process_frame(
            circuit_config,
            in_frame,
            effective_weights,
            membrane_voltage_offsets,
            diagnostics=True,
        )

        expected_spike_steps = np.full(100, -1)
        expected_spike_steps[[2, 4, 7, 9]] = 0

        assert_array_equal(result.out_frame, np.array([2, 4, 7, 9]))
        assert_array_equal(result.spike_steps, expected_spike_steps)

        assert result.step_membrane_voltages is not None

        assert_array_almost_equal(
            result.step_membrane_voltages[0], membrane_voltage_offsets + np.ones(100)
        )

    def test_sparse_propagation(self):
        circuit_config = CircuitConfig(
            N=100,
            compartments=[
                CompartmentConfig(neuron_indexes=slice(100), max_spikes_per_cycle=10)
            ],
        )

        in_frame = np.array([3, 8])

        effective_weights = np.zeros((circuit_config.N, circuit_config.N))
        effective_weights[11, 3] = 0.6
        effective_weights[11, 8] = 0.5

        effective_weights[15, 11] = 0.7
        effective_weights[15, 3] = 0.3

        membrane_voltage_offsets = np.zeros(circuit_config.N)

        result = process_frame(
            circuit_config,
            in_frame,
            effective_weights,
            membrane_voltage_offsets,
            diagnostics=True,
        )

        expected_spike_steps = np.full(100, -1)
        expected_spike_steps[[3, 8, 11, 15]] = [0, 0, 1, 2]

        expected_step_membrane_voltages = np.zeros((4, 100))
        expected_step_membrane_voltages[0, 3] = 1
        expected_step_membrane_voltages[0, 8] = 1
        expected_step_membrane_voltages[1, 11] = 1.1
        expected_step_membrane_voltages[1, 15] = 0.3
        expected_step_membrane_voltages[2, 15] = 1.0

        assert_array_equal(result.out_frame, np.array([3, 8, 11, 15]))
        assert_array_equal(result.spike_steps, expected_spike_steps)
        assert result.step_membrane_voltages is not None
        assert_array_almost_equal(
            result.step_membrane_voltages, expected_step_membrane_voltages
        )

    def test_compartments(self):
        circuit_config = CircuitConfig(
            N=100,
            compartments=[
                CompartmentConfig(neuron_indexes=slice(70), max_spikes_per_cycle=3),
                CompartmentConfig(
                    neuron_indexes=slice(70, 100), max_spikes_per_cycle=2
                ),
            ],
        )

        in_frame = np.arange(100)
        effective_weights = np.zeros((circuit_config.N, circuit_config.N))
        membrane_voltage_offsets = np.zeros(circuit_config.N)

        membrane_voltage_offsets[2] = 0.3
        membrane_voltage_offsets[4] = 0.2
        membrane_voltage_offsets[7] = 0.2
        membrane_voltage_offsets[9] = 0.1

        membrane_voltage_offsets[90] = 0.1
        membrane_voltage_offsets[91] = 0.1

        result = process_frame(
            circuit_config,
            in_frame,
            effective_weights,
            membrane_voltage_offsets,
            diagnostics=True,
        )

        expected_spike_steps = np.full(100, -1)
        expected_spike_steps[[2, 4, 7, 90, 91]] = 0

        expected_step_membrane_voltages = np.ones((2, 100)) + membrane_voltage_offsets
        expected_step_membrane_voltages[1, [2, 4, 7, 90, 91]] = 0.0

        assert_array_equal(result.out_frame, np.array([2, 4, 7, 90, 91]))
        assert_array_equal(result.spike_steps, expected_spike_steps)

        assert result.step_membrane_voltages is not None
        assert_array_almost_equal(
            result.step_membrane_voltages, expected_step_membrane_voltages
        )

    def test_complext_scenario(self):
        circuit_config = CircuitConfig(
            N=20,
            compartments=[
                CompartmentConfig(neuron_indexes=slice(5), max_spikes_per_cycle=10),
                CompartmentConfig(neuron_indexes=slice(5, 15), max_spikes_per_cycle=5),
                CompartmentConfig(neuron_indexes=slice(15, 20), max_spikes_per_cycle=2),
            ],
        )

        in_frame = np.array([0, 3])

        effective_weights = np.zeros((20, 20))
        effective_weights[5, 0] = 0.7
        effective_weights[5, 3] = 0.4
        effective_weights[5, 15] = 0.4
        effective_weights[7, 0] = 0.6
        effective_weights[7, 5] = 0.8

        effective_weights[14, 7] = 0.9

        effective_weights[15, 5] = 0.3
        effective_weights[15, 7] = 0.4
        effective_weights[15, 11] = 0.4

        effective_weights[12, 7] = 0.1
        effective_weights[8, 16] = 1.3
        effective_weights[12, 15] = 0.1
        effective_weights[12, 17] = 0.7
        effective_weights[12, 14] = 0.3
        effective_weights[13, 5] = 0.1
        effective_weights[13, 15] = 0.1
        effective_weights[13, 17] = 0.7
        effective_weights[13, 14] = 0.3
        effective_weights[14, 15] = 0.1
        effective_weights[14, 17] = 0.1

        effective_weights[16, 7] = 0.8
        effective_weights[16, 11] = 0.3
        effective_weights[17, 5] = 0.6
        effective_weights[17, 11] = 0.3

        membrane_voltage_offsets = np.zeros(20)
        membrane_voltage_offsets[11] = 1.1
        membrane_voltage_offsets[12] = -0.1
        membrane_voltage_offsets[16] = -0.1
        membrane_voltage_offsets[17] = 0.2

        result = process_frame(
            circuit_config,
            in_frame,
            effective_weights,
            membrane_voltage_offsets,
            diagnostics=True,
        )

        expected_spike_steps = np.full(20, -1)
        expected_spike_steps[0] = 0
        expected_spike_steps[3] = 0
        expected_spike_steps[11] = 0
        expected_spike_steps[5] = 1
        expected_spike_steps[7] = 2
        expected_spike_steps[17] = 2
        expected_spike_steps[15] = 3
        expected_spike_steps[13] = 4
        expected_spike_steps[14] = 3

        expected_step_membrane_voltages = np.zeros((6, 20))
        expected_step_membrane_voltages[:, [12, 16]] = [-0.1, -0.1]
        expected_step_membrane_voltages[0, 17] = 0.2

        expected_step_membrane_voltages[0, [0, 3, 11]] = [1.0, 1.0, 1.1]
        expected_step_membrane_voltages[1, [5, 7, 15, 16, 17]] = [
            1.1,
            0.6,
            0.4,
            0.2,
            0.5,
        ]

        expected_step_membrane_voltages[2, [7, 13, 15, 16, 17]] = [
            1.4,
            0.1,
            0.7,
            0.2,
            1.1,
        ]

        expected_step_membrane_voltages[3, [12, 13, 14, 15, 16]] = [
            0.7,
            0.8,
            1.0,
            1.1,
            1.0,
        ]

        expected_step_membrane_voltages[4, [12, 13, 16]] = [
            1.1,
            1.2,
            1.0,
        ]

        expected_step_membrane_voltages[5, [12, 16]] = [
            1.1,
            1.0,
        ]

        assert_array_equal(result.out_frame, np.array([0, 3, 5, 7, 11, 13, 14, 15, 17]))
        assert_array_almost_equal(result.spike_steps, expected_spike_steps)
        assert result.step_membrane_voltages is not None

        print(expected_spike_steps)

        assert_array_almost_equal(
            result.step_membrane_voltages, expected_step_membrane_voltages
        )
