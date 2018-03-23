from data.io import load_samples
from data.extract import get_input_variables, get_target_variable
from algorithm.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithm.semantic_learning_machine.mutation_operator import Mutation2
from algorithm.common.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
from algorithm.common.metric import RootMeanSquaredError, Accuracy
import unittest


class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        self.training, self.validation, self.testing = load_samples('c_cancer', 0)

    def test_fit(self):
        print("Basic tests of fit()...")
        algorithm = SemanticLearningMachine(100, MaxGenerationsCriterion(200), 3, 0.01, 50, Mutation2())
        X = get_input_variables(self.training).as_matrix()
        y = get_target_variable(self.training).as_matrix()
        algorithm.fit(X, y, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=algorithm.champion)
        print()

    def test_ols(self):
        print('OLS tests of fit()...')
        algorithm = SemanticLearningMachine(100, MaxGenerationsCriterion(200), 3, 'optimized', 50, Mutation2())
        X = get_input_variables(self.training).as_matrix()
        y = get_target_variable(self.training).as_matrix()
        algorithm.fit(X, y, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=algorithm.champion)
        print()

    def test_edv(self):
        print('EDV tests of fit()...')
        algorithm = SemanticLearningMachine(100, ErrorDeviationVariationCriterion(0.25), 3, 0.01, 50, Mutation2())
        X = get_input_variables(self.training).as_matrix()
        y = get_target_variable(self.training).as_matrix()
        algorithm.fit(X, y, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=algorithm.champion)
        print()

if __name__ == '__main__':
    unittest.main()
