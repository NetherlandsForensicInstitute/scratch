"""
Statistical utilities for the NIST CMC pipeline.
"""

import numpy as np
from scipy import stats as sp_stats


def detect_outliers_esd(
    data: np.ndarray, alpha: float = 0.05, max_outliers: int = 1
) -> np.ndarray:
    """
    Generalised Extreme Studentized Deviate (ESD) test for outliers.

    Faithful translation of ``stat_idout_esd.m`` (Rosner 1983).

    The algorithm iteratively removes the most extreme observation and
    compares the test statistic against the Rosner critical value.
    Only outliers up to the *last* iteration where H₀ was rejected are marked.

    :param data: 1-D array of observations (NaN values are ignored).
    :param alpha: Significance level (default 0.05).
    :param max_outliers: Upper bound on the number of outliers to test.
    :returns: Boolean array of the same shape as *data*, True = outlier.

    Reference:
        Rosner, B. (1983). Percentage Points for a Generalized ESD
        Many-Outlier Procedure. *Technometrics*, 25(2), 165–172.
    """

    def _rosner_critical(alpha_: float, n: np.ndarray) -> np.ndarray:
        n = np.asarray(n, dtype=float)
        p = 1.0 - alpha_ / (2.0 * (n + 1.0))
        t = sp_stats.t.ppf(p, n - 1.0)
        return t * n / np.sqrt((n - 1.0 + t**2) * (n + 1.0))

    x = np.asarray(data, dtype=float).ravel()
    n_total = x.size

    if max_outliers <= 0 or n_total < 3:
        return np.zeros(n_total, dtype=bool)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    valid = ~np.isnan(x_sorted)
    n_obs = int(np.sum(valid))

    if n_obs < 3:
        return np.zeros(n_total, dtype=bool)

    crit_values = _rosner_critical(alpha, np.arange(1, n_total + 1))
    first_valid = int(np.argmax(valid))
    last_valid = n_total - 1 - int(np.argmax(valid[::-1]))

    valid_data = x_sorted[valid]
    running_mean = np.mean(valid_data)
    running_ss = np.sum((valid_data - running_mean) ** 2)
    n_current = n_obs
    last_rejection = 0
    removed_indices = []

    for i in range(max_outliers):
        if n_current < 3:
            break

        dev_low = running_mean - x_sorted[first_valid]
        dev_high = x_sorted[last_valid] - running_mean

        if dev_low >= dev_high:
            removed_idx = first_valid
            max_dev = dev_low
            first_valid += 1
            while first_valid < n_total and np.isnan(x_sorted[first_valid]):
                first_valid += 1
        else:
            removed_idx = last_valid
            max_dev = dev_high
            last_valid -= 1
            while last_valid >= 0 and np.isnan(x_sorted[last_valid]):
                last_valid -= 1

        removed_indices.append(removed_idx)

        std = np.sqrt(running_ss / n_current) if n_current > 0 else 0.0
        test_stat = max_dev / std if std > 1e-15 else 0.0
        crit = crit_values[n_current - i - 1]
        if test_stat > crit:
            last_rejection = i + 1

        removed_val = x_sorted[removed_idx]
        n_current -= 1
        if n_current > 0:
            new_mean = running_mean + (running_mean - removed_val) / n_current
            running_ss -= (removed_val - running_mean) * (removed_val - new_mean)
            running_mean = new_mean
        else:
            running_ss = 0.0

    is_outlier_sorted = np.zeros(n_total, dtype=bool)
    for j in range(last_rejection):
        is_outlier_sorted[removed_indices[j]] = True

    is_outlier = np.zeros(n_total, dtype=bool)
    is_outlier[sort_idx] = is_outlier_sorted
    is_outlier[np.isnan(x)] = False

    return is_outlier
