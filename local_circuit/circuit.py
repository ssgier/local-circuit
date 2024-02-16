import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from local_circuit.circuit_config import CircuitConfig
from local_circuit.ltp import update_ltp_weights
from local_circuit.stp import update_stp_weights


@dataclass
class ProcessFrameResult:
    out_frame: np.ndarray
    spike_steps: np.ndarray
    step_membrane_voltages: Optional[np.ndarray]


def process_frame(
    circuit_config: CircuitConfig,
    in_frame: np.ndarray,
    effective_weights: np.ndarray,
    membrane_voltage_offsets: np.ndarray,
    diagnostics: bool = False,
) -> ProcessFrameResult:
    membrane_voltages_start = membrane_voltage_offsets.copy()

    if len(in_frame) > 0:
        membrane_voltages_start[in_frame] += 1.0

    return process_membrane_voltages(
        circuit_config, effective_weights, membrane_voltages_start, diagnostics
    )


def process_membrane_voltages(
    circuit_config: CircuitConfig,
    effective_weights: np.ndarray,
    membrane_voltages_start: np.ndarray,
    diagnostics: bool = False,
) -> ProcessFrameResult:
    neuron_ids = np.arange(circuit_config.N)

    # -1 means not spiked, e.g. 2 means spiked in step 2
    spike_steps = np.full(circuit_config.N, -1)
    dirty = True
    step = 0

    step_membrane_voltages: Optional[List[np.ndarray]] = [] if diagnostics else None

    while dirty:
        dirty = False
        spiked_mask = spike_steps != -1

        epsps = effective_weights[:, spiked_mask].sum(axis=1)
        membrane_voltages = np.where(
            spiked_mask, np.zeros(circuit_config.N), membrane_voltages_start + epsps
        )

        if step_membrane_voltages is not None:
            step_membrane_voltages.append(membrane_voltages)

        for compartment_conf in circuit_config.compartments:
            compartment_spike_count = sum(
                spike_steps[compartment_conf.neuron_indexes] != -1
            )
            spikes_left = (
                compartment_conf.max_spikes_per_cycle - compartment_spike_count
            )

            if spikes_left <= 0:
                continue

            compartment_v = membrane_voltages[compartment_conf.neuron_indexes]

            above_threshold_mask = compartment_v >= 1.0

            num_spikes = min(spikes_left, sum(above_threshold_mask))

            if sum(above_threshold_mask) > num_spikes:
                compartment_v_above_threshold = compartment_v[above_threshold_mask]
                spiked_subindexes = (-compartment_v_above_threshold).argpartition(
                    num_spikes
                )[:num_spikes]
                spiked_neuron_ids = neuron_ids[compartment_conf.neuron_indexes][
                    above_threshold_mask
                ][spiked_subindexes]
            else:
                spiked_neuron_ids = neuron_ids[compartment_conf.neuron_indexes][
                    above_threshold_mask
                ]
            if num_spikes > 0:
                spike_steps[spiked_neuron_ids] = step
                dirty = True
        step += 1

    out_frame = neuron_ids[spike_steps != -1]

    return ProcessFrameResult(
        out_frame=out_frame,
        spike_steps=spike_steps,
        step_membrane_voltages=np.array(step_membrane_voltages),
    )


class Circuit:
    def __init__(self, config: CircuitConfig) -> None:
        # TODO: add more extensive diversity, probably the first place to try would be the stp skew, as well as the stp time constant, next would be the stdp algorithm (non-causal potentiation, including reverse stepping)

        weights_shape = (config.N, config.N)
        self.t = 0
        self.weight_mask = np.ones(weights_shape)
        self.lt_weights = np.random.random(weights_shape)
        self.st_weights = np.zeros(weights_shape)
        self.window_rates = np.zeros(config.N)
        self.config = config
        self.shallow_recording: Optional[List[np.ndarray]] = None
        self.deep_recording: Optional[List[np.ndarray]] = None

    def process_cycle(
        self,
        in_frame: np.ndarray,
        tf_spike: np.ndarray = np.array([]),  # teacher forcing
        tf_no_spike: np.ndarray = np.array([]),
    ) -> np.ndarray:
        homeostasis_offsets = self._compute_homeostasis_offsets()
        stochastic_offsets = self._compute_stochastic_offsets()
        process_frame_result = process_frame(
            self.config,
            in_frame,
            self._get_effective_weights(),
            homeostasis_offsets + stochastic_offsets,
            diagnostics=False,
        )

        tf_adjusted_spike_steps = process_frame_result.spike_steps

        if len(tf_spike) > 0:
            tf_adjusted_spike_steps[tf_spike] = (
                process_frame_result.spike_steps.max() + 1
            )

        if len(tf_no_spike) > 0:
            tf_adjusted_spike_steps[tf_no_spike] = -1

        update_stp_weights(self.st_weights, tf_adjusted_spike_steps, self.config)

        update_ltp_weights(self.lt_weights, tf_adjusted_spike_steps, self.config)

        self._update_window_rates(process_frame_result.out_frame)

        if self.shallow_recording is not None:
            self.shallow_recording.append(process_frame_result.out_frame)

        if self.deep_recording is not None:
            self.deep_recording.append(process_frame_result.spike_steps)

        return process_frame_result.out_frame

    def advance_cycles(self, num_cycles: int) -> np.ndarray:
        out_frames = []
        for _ in range(num_cycles):
            out_frames.append(self.process_cycle(np.array([])))

        return np.array(out_frames)

    # out frames
    def start_recording_shallow(self):
        self.shallow_recording = []

    def stop_recording_shallow(self) -> List[np.ndarray]:
        if self.shallow_recording is None:
            raise Exception("not recording")

        ret_val = self.shallow_recording
        self.shallow_recording = None
        return ret_val

    # steps
    def start_recording_deep(self):
        self.deep_recording = []

    def stop_recording_deep(self) -> np.ndarray:
        if self.deep_recording is None:
            raise Exception("not recording")

        ret_val = np.array(self.deep_recording)
        self.deep_recording = None
        return ret_val

    def _update_window_rates(self, out_frame: np.ndarray):
        decay_factor = self.config.window_rate_decay_factor
        updates = np.zeros(self.config.N)
        updates[out_frame] = 1.0
        self.window_rates = (
            decay_factor * self.window_rates + (1 - decay_factor) * updates
        )

    def _get_effective_weights(self) -> np.ndarray:
        stp_skew = self.config.stp_skew_factor
        scale = self.config.weight_scale_factor
        return scale * (stp_skew * self.st_weights + (1 - stp_skew) * self.lt_weights)

    def _compute_homeostasis_offsets(self) -> np.ndarray:
        return (
            self.config.homeostasis_target - self.window_rates
        ) * self.config.homeostasis_scale_factor

    def _compute_stochastic_offsets(self) -> np.ndarray:
        return (
            np.random.random(self.config.N) - 0.5
        ) * self.config.stochastic_offset_range
