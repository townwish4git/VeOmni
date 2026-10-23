# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Timing statistics and shard geometry for the dummy Muon training entry."""

import math
import statistics
from typing import Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    """Compute a linearly interpolated percentile, including a one-sample window."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Percentiles require samples and a fraction in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def timing_stats(values: Sequence[float]) -> dict[str, float]:
    """Summarize milliseconds without dropping outliers."""
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timings must be finite, positive, and nonempty")
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": deviation,
        "cv": deviation / mean,
    }


def stable_window(samples: Sequence[float], window: int, max_cv: float, max_drift: float) -> bool:
    """Require two adjacent windows to have low variation and low median drift."""
    if window < 2 or len(samples) < 2 * window:
        return False
    previous, current = samples[-2 * window : -window], samples[-window:]
    before, after = timing_stats(previous), timing_stats(current)
    drift = abs(after["median_ms"] / before["median_ms"] - 1)
    return max(before["cv"], after["cv"]) <= max_cv and drift <= max_drift


def local_shard(shape: Sequence[int], shards: int, rank: int, dimension: int = 0) -> tuple[list[int], int]:
    """Match Torch Shard's ceil-chunk layout, including empty trailing shards."""
    if not shape or shards < 1 or not 0 <= rank < shards:
        raise ValueError("Invalid shard shape, shard count, or rank")
    dimension %= len(shape)
    chunk = math.ceil(shape[dimension] / shards)
    offset = min(rank * chunk, shape[dimension])
    result = list(shape)
    result[dimension] = min(chunk, shape[dimension] - offset)
    return result, offset
