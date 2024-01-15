import numpy as np
from local_circuit.circuit_config import CircuitConfig


def update_stp_weights(stp_weights: np.ndarray, spike_steps: np.ndarray, circuit_config: CircuitConfig) -> None:

    stp_weights *= circuit_config.stp_decay_factor

    update_mask = spike_steps != -1
    col = spike_steps.reshape(circuit_config.N, 1)
    row = col.T

    stp_weights[update_mask] = np.where(
        (col[update_mask] > row) & update_mask, 1., 0.)
