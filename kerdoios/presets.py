from __future__ import annotations

from .types import Mode

# cost, capability, latency, throughput — higher weight = more important
PRESETS: dict[Mode, dict[str, float]] = {
    Mode.FREE: {"cost": 1.0, "capability": 0.35, "latency": 0.1, "throughput": 0.2},
    Mode.CHEAP: {"cost": 0.9, "capability": 0.65, "latency": 0.2, "throughput": 0.35},
    Mode.BALANCED: {"cost": 0.7, "capability": 0.8, "latency": 0.4, "throughput": 0.5},
    Mode.FAST: {"cost": 0.25, "capability": 0.55, "latency": 1.0, "throughput": 0.6},
    Mode.MAX: {"cost": 0.2, "capability": 1.0, "latency": 0.3, "throughput": 0.3},
    Mode.SCALE: {"cost": 0.35, "capability": 0.55, "latency": 0.3, "throughput": 1.0},
    Mode.PRIVATE: {"cost": 0.5, "capability": 0.7, "latency": 0.3, "throughput": 0.3},
}
