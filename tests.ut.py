import unittest

from tests import sign, wilcoxon, friedman, nemenyi


class TestCase(unittest.TestCase):
    def test_sign(self):
        # example from: https://en.wikipedia.org/wiki/Sign_test

        x = [142, 140, 144, 144, 142, 146, 149, 150, 142, 148]
        y = [138, 136, 147, 139, 143, 141, 143, 145, 136, 146]

        p = sign(x, y)

        self.assertEqual(p, 0.109375)

    def test_wilcoxon(self):
        # example from: http://www.statstutor.ac.uk/resources/uploaded/wilcoxonsignedranktest.pdf

        x = [2.0, 3.6, 2.6, 2.6, 7.3, 3.4, 14.9, 6.6, 2.3, 2.0, 6.8, 8.5]
        y = [3.5, 5.7, 2.9, 2.4, 9.9, 3.3, 16.7, 6.0, 3.8, 4.0, 9.1, 20.9]

        test_statistic, p = wilcoxon(x, y)

        self.assertEqual(test_statistic, 7.0)
        self.assertEqual(round(p, 3), 0.012)

    def test_friedman(self):
        # example from: Japkowicz, Nathalie, and Mohak Shah. Evaluating learning algorithms:
        # a classification perspective. Cambridge University Press, 2011.

        a = [85.83, 85.91, 86.12, 85.82, 86.28, 86.42, 85.91, 86.10, 85.95, 86.12]
        b = [75.86, 73.18, 69.08, 74.05, 74.71, 65.90, 76.25, 75.10, 70.50, 73.95]
        c = [84.19, 85.91, 83.83, 85.11, 86.38, 81.20, 86.38, 86.75, 88.03, 87.18]

        with self.assertWarns(Warning):
            test_statistic, p, average_ranks = friedman([a, b, c])

        self.assertEqual(round(test_statistic, 2), 15.05)
        self.assertEqual(average_ranks, [1.55, 3.0, 1.45])

    def test_nemenyi(self):
        pass


if __name__ == '__main__':
    unittest.main()
