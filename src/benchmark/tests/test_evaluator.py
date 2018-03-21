from algorithm.common.metric import RootMeanSquaredError
from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorSGA
from benchmark.configuration import SLM_OLS_CONFIGURATIONS, NEAT_CONFIGURATIONS, SGA_CONFIGURATIONS
from data.io_data_set import load_samples
import unittest

class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        self.training, self.validation, self.testing = load_samples('c_cancer', 0)

    def test_evaluator_slm(self):
        evaluator_slm = EvaluatorSLM(SLM_OLS_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        # learner_meta = evaluator_slm.run(time_limit=10, time_buffer=0, verbose=False)
        # self.assertTrue(expr=learner_meta)

    def test_evaluator_neat(self):
        evaluator_neat = EvaluatorNEAT(NEAT_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        # learner_meta = evaluator_neat.run(time_limit=60, time_buffer=0.1, verbose=True)
        # self.assertTrue(expr=learner_meta)

    def test_evaluator_sga(self):
        evaluator_sga = EvaluatorSGA(SGA_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_sga.run(time_limit=30, time_buffer=0.1, verbose=True)
        self.assertTrue(expr=learner_meta)

if __name__ == '__main__':
    unittest.main()
