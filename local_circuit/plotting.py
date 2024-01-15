import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from local_circuit.circuit_config import CircuitConfig


def plot_shallow_recording(
    recording: List[np.ndarray], circuit_config: CircuitConfig
) -> plt.Figure:
    num_spikes = sum(len(frame) for frame in recording)
    cycles = np.empty(num_spikes)

    cycles_pos = 0
    for cycle, frame in enumerate(recording):
        next_cycles_pos = cycles_pos + len(frame)
        cycles[cycles_pos:next_cycles_pos] = cycle
        cycles_pos = next_cycles_pos

    neuron_ids = np.concatenate(recording)

    compartment_boundaries = [
        compartment.neuron_indexes.stop
        for compartment in circuit_config.compartments[0:-1]
    ]

    fig, ax = plt.subplots()
    ax.set_ylim([0, circuit_config.N])
    ax.scatter(cycles, neuron_ids, s=3, color="black")

    for cb in compartment_boundaries:
        ax.axhline(cb, color="black", lw=0.2)

    ax.set_xlabel("cycle")
    ax.set_ylabel("neuron id")
    return fig


def plot_deep_recording(
    recording: np.ndarray, circuit_config: CircuitConfig
) -> plt.Figure:
    all_neuron_ids = np.arange(circuit_config.N)
    cycle_array_parts = []
    neuron_id_array_parts = []
    grayscale_array_parts = []

    for cycle, spike_steps in enumerate(recording):
        spike_mask = spike_steps != -1
        spike_steps_spiking = spike_steps[spike_mask]
        if len(spike_steps_spiking) > 0:
            grayscale_values = 1.0 - spike_steps_spiking / spike_steps_spiking.max()

            cycle_array_parts.append(np.full_like(spike_steps_spiking, cycle))
            neuron_id_array_parts.append(all_neuron_ids[spike_mask])
            grayscale_array_parts.append(grayscale_values)

    cycles = np.concatenate(cycle_array_parts)
    neuron_ids = np.concatenate(neuron_id_array_parts)
    grayscale_values = np.concatenate(grayscale_array_parts)

    print(cycles)
    print(neuron_ids)
    print(grayscale_values)

    fig, ax = plt.subplots()
    ax.set_ylim([0, circuit_config.N])
    ax.scatter(
        cycles,
        neuron_ids,
        c=grayscale_values,
        cmap="YlOrBr",
        s=20,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    compartment_boundaries = [
        compartment.neuron_indexes.stop
        for compartment in circuit_config.compartments[0:-1]
    ]

    for cb in compartment_boundaries:
        ax.axhline(cb, color="black", lw=0.2)

    ax.set_xlabel("cycle")
    ax.set_ylabel("neuron id")
    return fig


def plot_weights_heatmap(
    weights: np.ndarray,
    circuit_config: CircuitConfig,
    weight_mask: Optional[np.ndarray] = None,
) -> plt.Figure:
    data = weights if weight_mask is None else weights * weight_mask
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="cividis")

    compartment_boundaries = [
        compartment.neuron_indexes.stop
        for compartment in circuit_config.compartments[0:-1]
    ]

    for cb in compartment_boundaries:
        ax.axvline(cb, color="black", lw=1, alpha=0.5)
        ax.axhline(cb, color="black", lw=1, alpha=0.5)

    return fig
