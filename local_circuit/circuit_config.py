from typing import List
from dataclasses import dataclass, field


@dataclass
class CompartmentConfig:
    neuron_indexes: slice
    max_spikes_per_cycle: int = 10


@dataclass
class CircuitConfig:
    N: int = 1000
    max_connect_density: float = 0.2
    ltp_step_up: float = 0.25
    ltp_step_down: float = 0.35
    stp_decay_factor: float = 0.8
    stp_skew_factor: float = 0.75
    weight_scale_factor: float = 0.25
    stochastic_offset_range: float = 0.0
    window_rate_decay_factor: float = 0.999
    homeostasis_target: float = 0.02
    homeostasis_scale_factor: float = 0.0
    compartments: List[CompartmentConfig] = field(default_factory=lambda: [])

    def __post_init__(self):
        if len(self.compartments) == 0:
            self.compartments = [CompartmentConfig(neuron_indexes=slice(self.N))]
