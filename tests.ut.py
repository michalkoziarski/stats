import unittest

from tests import sign, wilcoxon, friedman, nemenyi


class TestCase(unittest.TestCase):
    def test_sign(self):
        # example from https://en.wikipedia.org/wiki/Sign_test

        x = [142, 140, 144, 144, 142, 146, 149, 150, 142, 148]
        y = [138, 136, 147, 139, 143, 141, 143, 145, 136, 146]

        p = sign(x, y)

        self.assertEqual(p, 0.109375)

    def test_wilcoxon(self):
        # example from http://www.statstutor.ac.uk/resources/uploaded/wilcoxonsignedranktest.pdf

        x = [2.0, 3.6, 2.6, 2.6, 7.3, 3.4, 14.9, 6.6, 2.3, 2.0, 6.8, 8.5]
        y = [3.5, 5.7, 2.9, 2.4, 9.9, 3.3, 16.7, 6.0, 3.8, 4.0, 9.1, 20.9]

        test_statistic, p = wilcoxon(x, y)

        self.assertEqual(test_statistic, 7.0)
        self.assertEqual(round(p, 3), 0.012)

    def test_friedman(self):
        pass

    def test_nemenyi(self):
        pass


if __name__ == '__main__':
    unittest.main()
