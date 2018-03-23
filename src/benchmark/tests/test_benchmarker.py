from benchmark.benchmarker import Benchmarker, continue_benchmark
import unittest

class TestAlgorithm(unittest.TestCase):

    # def setUp(self):
    #     self.benchmarker = Benchmarker('r_concrete')
    #
    # def test_run(self):
    #     self.benchmarker.run()

    def test_continue_benchmark(self):
        benchmarker = continue_benchmark('r_concrete', 'r_concrete__2018_03_23__17_44_55.pkl')


if __name__ == '__main__':
    unittest.main()