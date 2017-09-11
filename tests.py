from __future__ import division
from __future__ import print_function

import warnings

from math import floor, sqrt
from scipy.special import binom, chdtrc
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

    # sum the probability of suitable events (one-sided) using the binomial test
    for i in range(floor(min(n_x, n_y)) + 1):
        p += binom(n, i) * 0.5 ** n

    # adjust for the two-sidedness
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

    for x_i, y_i in zip(x, y):
        if x_i != y_i:
            if x_i > y_i:
                signs.append(1)
            else:
                signs.append(-1)

            absolute_differences.append(abs(x_i - y_i))

    n = len(signs)  # reduced sample size

    if n < 10:
        warnings.warn('Warning: sample size after the reduction is too small (should be >= 10, is %d).' % n)

    ranks = _rank(absolute_differences, order='desc')

    positive_rank_sum = sum([ranks[i] for i in range(n) if signs[i] == 1])
    negative_rank_sum = sum([ranks[i] for i in range(n) if signs[i] == -1])

    test_statistic = min(positive_rank_sum, negative_rank_sum)

    # calculate the z statistic
    mean = n * (n + 1) / 4.0
    std_dev = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z_statistic = (test_statistic - mean) / std_dev

    # calculate the p-value based on the survival function
    p = 2.0 * norm.sf(abs(z_statistic))

    return test_statistic, p


def friedman(x):
    """
    The Friedman test.

    Arguments:
        x: matrix of measurements, with x[i][j] being the measurement for the i-th method on the j-th domain

    Returns:
        Test statistic, associated p-value and average ranks.
    """

    k = len(x)  # the number of evaluated methods

    for i in range(k - 1):
        assert len(x[i]) == len(x[i + 1])

    n = len(x[0])  # the number of domains

    if n <= 15 or k <= 4:
        warnings.warn('Warning: the number of evaluated methods or the number of domains is too small '
                      '(should be > 15 and > 4, respectively, is %d and %d).' % (n, k))

    ranks = []

    for j in range(n):
        row = []

        for i in range(k):
            row.append(x[i][j])

        ranks.append(_rank(row, order='asc'))

    average_ranks = []  # of methods on all domains

    for i in range(k):
        average_ranks.append(sum([ranks[j][i] for j in range(n)]) / n)

    test_statistic = 12 / (n * k * (k + 1)) * sum([(avg_rank * n) ** 2 for avg_rank in average_ranks]) - 3 * n * (k + 1)

    # calculate the p-value based on the survival function
    p = 2.0 * chdtrc(k - 1, test_statistic)

    return test_statistic, p, average_ranks


def nemenyi(x):
    """
    The Nemenyi post-hoc test.

    Arguments:
        x: matrix of measurements
    """
    pass


def _rank(x, order):
    """
    Compute the ranks of the measurements.

    Arguments:
        x: list of measurements
        order: order of ranking, either 'asc' (the highest value will have the highest rank) or 'desc' (the opposite)

    Returns:
        A list of ranks.
    """

    assert order in ['asc', 'desc']

    ranks = []

    for i in range(len(x)):
        rank = 1

        for j in range(len(x)):
            if (order == 'asc' and x[i] < x[j]) or (order == 'desc' and x[i] > x[j]):
                rank += 1

        ranks.append(rank)

    # resolve ties
    for rank in range(1, max(ranks) + 1):
        count = 0
        indices = []

        for i in range(len(x)):
            if ranks[i] == rank:
                count += 1
                indices.append(i)

        for index in indices:
            ranks[index] = rank + 0.5 * (count - 1)

    return ranks
