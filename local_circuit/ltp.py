from local_circuit.circuit_config import CircuitConfig
import numpy as np


def update_ltp_weights(ltp_weights: np.ndarray, spike_steps: np.ndarray, circuit_config: CircuitConfig) -> None:
    # TODO: consider various modes, e.g. non-causal potentiation
    update_mask = spike_steps != -1

    col = spike_steps.reshape(circuit_config.N, 1)
    row = col.T

    diag_original = ltp_weights.diagonal().copy()

    ltp_weights[:, update_mask] += np.where(row[:, update_mask] <
                                            col, circuit_config.ltp_step_up, -circuit_config.ltp_step_down)

    ltp_weights[:, update_mask] = np.clip(
        ltp_weights[:, update_mask], 0.0, 1.0)
    np.fill_diagonal(ltp_weights, diag_original)
