from benchmark.benchmarker import Benchmarker
import unittest

class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        self.benchmarker = Benchmarker('r_concrete')

    def test_run(self):
        self.benchmarker.run()

if __name__ == '__main__':
    unittest.main()
