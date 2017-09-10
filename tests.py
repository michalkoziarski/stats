import warnings

from math import floor, sqrt
from scipy.special import binom
from scipy.stats.distributions import norm


def sign(x, y):
    """
    The sign test.

    Arguments:
        x: first list of measurements
        y: second list of measurements

    Returns:
        The p-value of the hypothesis test.
    """

    assert len(x) == len(y)

    n_x = 0.0  # the number of times x outperforms y
    n_y = 0.0  # the number of times y outperforms x

    for x_i, y_i in zip(x, y):
        if x_i > y_i:
            n_x += 1
        elif y_i > x_i:
            n_y += 1
        else:
            n_x += 0.5
            n_y += 0.5

    n = len(x)
    p = 0.0

    for i in range(floor(min(n_x, n_y)) + 1):
        p += binom(n, i) * 0.5 ** n

    p *= 2.0

    return p


def wilcoxon(x, y):
    """
    The Wilcoxon signed-rank test.

    Arguments:
        x: first list of measurements
        y: second list of measurements

    Returns:
        Test statistic and the associated p-value.
    """

    assert len(x) == len(y)

    signs = []
    absolute_differences = []
    ranks = []

    for x_i, y_i in zip(x, y):
        if x_i != y_i:
            if x_i > y_i:
                signs.append(1)
            else:
                signs.append(-1)

            absolute_differences.append(abs(x_i - y_i))

    n = len(signs)  # reduced sample size

    if n < 10:
        warnings.warn('Warning: sample size too small.')

    for i in range(n):
        rank = 1

        for j in range(n):
            if absolute_differences[i] > absolute_differences[j]:
                rank += 1

        ranks.append(rank)

    # resolve ties
    for rank in range(max(ranks)):
        count = 0
        indices = []

        for i in range(n):
            if ranks[i] == rank:
                count += 1
                indices.append(i)

        for index in indices:
            ranks[index] = rank + 0.5 * (count - 1)

    positive_rank_sum = sum([ranks[i] for i in range(n) if signs[i] == 1])
    negative_rank_sum = sum([ranks[i] for i in range(n) if signs[i] == -1])

    test_statistic = min(positive_rank_sum, negative_rank_sum)

    mean = n * (n + 1) / 4.0
    std_dev = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z_statistic = (test_statistic - mean) / std_dev
    p = 2.0 * norm.sf(abs(z_statistic))  # the survival function

    return test_statistic, p


def friedman(x):
    """
    The Friedman test.

    Arguments:
        x: matrix of measurements
    """
    pass


def nemenyi(x):
    """
    The Nemenyi post-hoc test.

    Arguments:
        x: matrix of measurements
    """
    pass
