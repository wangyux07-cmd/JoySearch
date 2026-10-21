import math

import numpy as np


class AllocationPolicy:
    """A finite-arm policy that only sees feedback from its chosen actions."""

    def __init__(self, kind, arm_count, fixed_arm=0, exploration_rate=0.1):
        if kind not in {"fixed", "fixed_exploration", "ucb1"}:
            raise ValueError(f"Unknown allocation policy: {kind}")
        if not 0 <= fixed_arm < arm_count or not 0 <= exploration_rate <= 1:
            raise ValueError("Invalid fixed arm or exploration rate")
        self.kind, self.fixed_arm, self.exploration_rate = kind, fixed_arm, exploration_rate
        self.counts = np.zeros(arm_count, dtype=np.int64)
        self.rewards = np.zeros(arm_count, dtype=np.float64)

    def choose(self, rng):
        if self.kind == "fixed":
            return self.fixed_arm
        if self.kind == "fixed_exploration":
            return (
                int(rng.integers(len(self.counts)))
                if rng.random() < self.exploration_rate
                else self.fixed_arm
            )
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(untried[0])
        scores = self.rewards / self.counts + np.sqrt(
            2 * math.log(int(self.counts.sum())) / self.counts
        )
        return int(np.argmax(scores))

    def update(self, arm, reward):
        if not 0 <= reward <= 1:
            raise ValueError("Proxy reward must lie in [0, 1]")
        self.counts[arm] += 1
        self.rewards[arm] += reward
