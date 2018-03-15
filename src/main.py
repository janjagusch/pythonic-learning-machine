from src.data.io_data_set import load_samples
from semantic_learning_machine.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
from semantic_learning_machine.semantic_learning_machine import SemanticLearningMachine
from neat_python.neat_python import Neat
from benchmarker.parameter_tuner import SLM_CONFIGURATIONS, NEAT_CONFIGURATIONS, SGA_CONFIGURATION
from simple_genetic_algorithm.algorithm import SimpleGeneticAlgorithm
from simple_genetic_algorithm.neural_network.neural_network import create_network_from_topology
from simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from simple_genetic_algorithm.selection_operator import SelectionOperatorTournament

training, validation, testing = load_samples('r_concrete', 0)

semantic_learning_machine = SemanticLearningMachine(training_set=training, **SLM_CONFIGURATIONS[0])
neat = Neat(training_set=training, **NEAT_CONFIGURATIONS[0])
simple_genetic_algorithm = SimpleGeneticAlgorithm(training_set=training, **SGA_CONFIGURATION[0])

# semantic_learning_machine.run()
# neat.run()
simple_genetic_algorithm.run()