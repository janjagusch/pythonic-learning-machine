from src.data.io_data_set import load_samples
from algorithms.semantic_learning_machine.semantic_learning_machine import SemanticLearningMachine
from algorithms.neat_python.neat_python import Neat
from benchmark_experiments.parameter_tuner import SLM_FLS_CONFIGURATIONS, NEAT_CONFIGURATIONS, SGA_CONFIGURATIONS, SVR_CONFIGURATIONS, \
    MLP_CONFIGURATIONS
from algorithms.simple_genetic_algorithm.algorithm import SimpleGeneticAlgorithm
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from data.data_set import get_target_variable, get_input_variables
import numpy as np

training, validation, testing = load_samples('r_concrete', 0)

semantic_learning_machine = SemanticLearningMachine(**SLM_FLS_CONFIGURATIONS[0])
neat = Neat(training_set=training, **NEAT_CONFIGURATIONS[0])
# simple_genetic_algorithm = SimpleGeneticAlgorithm(training_set=training, **SGA_CONFIGURATIONS[0])
svr = SVR(**SVR_CONFIGURATIONS[0])
mlr = MLPRegressor(**MLP_CONFIGURATIONS[0])

semantic_learning_machine.fit(get_input_variables(training).as_matrix(), get_target_variable(training).as_matrix())
print(semantic_learning_machine.predict(get_input_variables(validation).as_matrix()))
# neat.run()
# simple_genetic_algorithm.run()
# svr.fit(get_input_variables(training).as_matrix(), get_target_variable(training).as_matrix())
# mlr.fit(get_input_variables(training).as_matrix(), get_target_variable(training).as_matrix())
