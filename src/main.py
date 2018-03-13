from src.data.io_data_set import load_samples
from semantic_learning_machine.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
from semantic_learning_machine.semantic_learning_machine import SemanticLearningMachine
from semantic_learning_machine.mutation_operators import Mutation2
from neat_python.neat_python import Neat
from benchmarker.parameter_tuner import SLM_CONFIGURATIONS, NEAT_CONFIGURATIONS


training, validation, testing = load_samples('r_concrete', 0)

semantic_learning_machine = SemanticLearningMachine(training_set=training, **SLM_CONFIGURATIONS[0])

neat = Neat(training_set=training, **NEAT_CONFIGURATIONS[0])

# stopping_criterion = ErrorDeviationVariationCriterion(0.25)
#
# semantic_learning_machine = SemanticLearningMachine(stopping_criterion, 500, 3,
#                                                     'optimized', 50, Mutation2(),
#                                                     training, validation, testing, True)

# semantic_learning_machine.run()

neat = Neat(training, 200, 100, 4, 1, 1, 0.25, 0.1, 0.25, 0.1)
neat.run()