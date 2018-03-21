from data.io_data_set import load_samples
from data.data_set import get_input_variables, get_target_variable
from benchmark.algorithm import BenchmarkSLM, BenchmarkNEAT, BenchmarkSGA
from algorithm.semantic_learning_machine.mutation_operators import Mutation2
from algorithm.common.stopping_criterion import MaxGenerationsCriterion
from algorithm.common.metric import RootMeanSquaredError, Accuracy
from algorithm.common.neural_network.neural_network import create_network_from_topology
from algorithm.simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from algorithm.simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from algorithm.simple_genetic_algorithm.selection_operator import SelectionOperatorTournament
import unittest


class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        self.training, self.validation, self.testing = load_samples('c_cancer', 0)

    def test_benchmark_slm(self):
        print('Test BenchmarkSLM()...')
        algorithm = BenchmarkSLM(10, MaxGenerationsCriterion(10), 3, 0.01, 50, Mutation2())
        X = get_input_variables(self.training).as_matrix()
        y = get_target_variable(self.training).as_matrix()
        log = algorithm.fit(X, y, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=log)
        print()

    def test_benchmark_neat(self):
        print('Test BenchmarkNEAT()...')
        algorithm = BenchmarkNEAT(10, MaxGenerationsCriterion(10), 4, 1, 1,
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        X = get_input_variables(self.training).as_matrix()
        y = get_target_variable(self.training).as_matrix()
        log = algorithm.fit(X, y, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=log)
        print()

    def test_benchmark_sga(self):
        print('Test BenchmarkSGA()...')
        topology = create_network_from_topology([2, 2])
        algorithm = BenchmarkSGA(10, MaxGenerationsCriterion(10), topology, SelectionOperatorTournament(5),
                                 MutationOperatorGaussian(0.1), CrossoverOperatorArithmetic(), 0.01, 0.25)
        X = get_input_variables(self.training).as_matrix()
        y = get_target_variable(self.training).as_matrix()
        log = algorithm.fit(X, y, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=log)
        print()

if __name__ == '__main__':
    unittest.main()
