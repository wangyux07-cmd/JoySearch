from collections import Counter

import numpy as np

from joysearch.traffic.policy import AllocationPolicy


def list_statistics(lists, frequencies, tail_max):
    counts = Counter(item for ranking in lists for item in ranking[:10])
    total = sum(counts.values())
    return {
        "catalog_coverage_at_10": len(counts) / len(frequencies),
        "tail_list_share_at_10": sum(
            n for item, n in counts.items() if frequencies[item] <= tail_max
        )
        / total
        if total
        else 0.0,
        "list_concentration_hhi": sum((n / total) ** 2 for n in counts.values()) if total else 0.0,
        "mean_train_frequency_at_10": sum(n * frequencies[item] for item, n in counts.items())
        / total
        if total
        else 0.0,
        "offline_list_appearances": total,
    }


def simulate(aliases, rewards, rankings, arms, fixed_arm, config, frequencies, tail_max):
    """Rating-target potential outcomes are diagnostic proxies, not exposure-log replay."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.shape != (len(aliases), len(arms)) or not np.isin(rewards, [0, 1]).all():
        raise ValueError("Simulation requires one binary potential outcome per request and arm")
    summaries, logs = [], []
    for seed in config["seeds"]:
        order = np.random.default_rng(seed).permutation(len(aliases))
        for kind in config["policies"]:
            policy = AllocationPolicy(kind, len(arms), fixed_arm, config["exploration_rate"])
            rng = np.random.default_rng(seed + 1000003)
            chosen_lists, cumulative, oracle = [], 0.0, 0.0
            for step, index in enumerate(order, 1):
                arm = policy.choose(rng)
                previous_count = int(policy.counts[arm])
                previous_mean = (
                    float(policy.rewards[arm] / previous_count) if previous_count else None
                )
                observed = float(rewards[index, arm])
                policy.update(arm, observed)
                cumulative += observed
                oracle += float(rewards[index].max())
                chosen_lists.append(rankings[index][arm])
                logs.append(
                    {
                        "seed": seed,
                        "policy": kind,
                        "step": step,
                        "user_alias": aliases[index],
                        "arm": arms[arm],
                        "observed_proxy_reward": observed,
                        "selected_arm_previous_count": previous_count,
                        "selected_arm_previous_mean": previous_mean,
                        "cumulative_proxy_reward": cumulative,
                        "oracle_diagnostic_cumulative_reward": oracle,
                    }
                )
            summaries.append(
                {
                    "seed": seed,
                    "policy": kind,
                    "sample_count": len(aliases),
                    "proxy_hit_rate": cumulative / len(aliases),
                    "proxy_reward_sum": cumulative,
                    "contextual_proxy_oracle_reward_sum": oracle,
                    "contextual_proxy_regret": oracle - cumulative,
                    "arm_counts": dict(zip(arms, policy.counts.tolist())),
                    **list_statistics(chosen_lists, frequencies, tail_max),
                }
            )
    return summaries, logs
