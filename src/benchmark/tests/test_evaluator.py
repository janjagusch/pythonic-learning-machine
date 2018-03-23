from algorithm.common.metric import RootMeanSquaredError
from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorSGA, EvaluatorSVC, \
    EvaluatorSVR, EvaluatorMLPC, EvaluatorMLPR, EvaluatorRFC, EvaluatorRFR
from benchmark.configuration import SLM_OLS_CONFIGURATIONS, NEAT_CONFIGURATIONS, SGA_CONFIGURATIONS, SVC_CONFIGURATIONS, \
    SVR_CONFIGURATIONS, MLP_CONFIGURATIONS, RF_CONFIGURATIONS
from data.io import load_samples
import unittest

class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        self.training, self.validation, self.testing = load_samples('c_cancer', 0)

    def test_evaluator_slm(self):
        evaluator_slm = EvaluatorSLM(SLM_OLS_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_slm.run(time_limit=10, time_buffer=0, verbose=False)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_neat(self):
        evaluator_neat = EvaluatorNEAT(NEAT_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_neat.run(time_limit=10, time_buffer=0.1, verbose=False)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_sga(self):
        evaluator_sga = EvaluatorSGA(SGA_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_sga.run(time_limit=10, time_buffer=0.1, verbose=False)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_svc(self):
        evaluator_svc = EvaluatorSVC(SVC_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_svc.run(time_limit=5, time_buffer=0.1)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_svr(self):
        evaluator_svr = EvaluatorSVR(SVR_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_svr.run(time_limit=5, time_buffer=0.1)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_mlpc(self):
        evaluator_mlpc = EvaluatorMLPC(MLP_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_mlpc.run(time_limit=5, time_buffer=0.1)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_mlpr(self):
        evaluator_mlpr = EvaluatorMLPR(MLP_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_mlpr.run(time_limit=5, time_buffer=0.1)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_rfc(self):
        evaluator_rfc = EvaluatorRFC(RF_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_rfc.run(time_limit=10, time_buffer=0.1)
        self.assertTrue(expr=learner_meta)

    def test_evaluator_rfr(self):
        evaluator_rfc = EvaluatorRFR(RF_CONFIGURATIONS, self.training, self.validation, self.testing, RootMeanSquaredError)
        learner_meta = evaluator_rfc.run(time_limit=10, time_buffer=0.1)
        self.assertTrue(expr=learner_meta)


if __name__ == '__main__':
    unittest.main()
