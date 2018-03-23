from benchmark.configuration import ENSEMBLE_CONFIGURATIONS
from data.io import load_samples
import unittest

class TestAlgorithm(unittest.TestCase):

    def test_ensemble_configurations(self):
        self.assertTrue(ENSEMBLE_CONFIGURATIONS.__class__ == list)


if __name__ == '__main__':
    unittest.main()
