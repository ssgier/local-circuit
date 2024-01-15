from local_circuit import circuit_config
from local_circuit.plotting import plot_shallow_recording
from local_circuit import circuit
from local_circuit.circuit_config import CircuitConfig, CompartmentConfig
from local_circuit.circuit import Circuit, process_frame
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

circuit_config = CircuitConfig(
    N=1000,
    compartments=[
        CompartmentConfig(neuron_indexes=slice(980), max_spikes_per_cycle=10),
        CompartmentConfig(neuron_indexes=slice(980, 990), max_spikes_per_cycle=2),
        CompartmentConfig(neuron_indexes=slice(990, 1000), max_spikes_per_cycle=3),
    ],
)

in_frame = np.array([0, 1, 2, 3, 4, 994])

result = process_frame(
    circuit_config,
    in_frame,
    np.random.random((circuit_config.N, circuit_config.N)),
    np.zeros(circuit_config.N),
)

cc = CircuitConfig(
    N=4,
    homeostasis_target=0.2,
    homeostasis_scale_factor=1 / 0.1,
    stochastic_offset_range=0.9,
    weight_scale_factor=0.0,
)

circuit = Circuit(cc)

circuit.start_recording_shallow()
for _ in range(1000):
    circuit.advance_cycles(1)
    print(f"rates: {circuit.window_rates}")


recording = circuit.stop_recording_shallow()
plot_shallow_recording(recording, cc)
plt.show()
