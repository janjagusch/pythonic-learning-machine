from src.data.io_data_set import load_samples
from semantic_learning_machine.stopping_criterion import MaxGenerationsCriterion
from semantic_learning_machine.semantic_learning_machine import SemanticLearningMachine
from semantic_learning_machine.mutation_operators import Mutation2


training, validation, testing = load_samples('r_concrete', 0)

stopping_criterion = MaxGenerationsCriterion(max_generation=500)

semantic_learning_machine = SemanticLearningMachine(stopping_criterion, 500, 3,
                                                    'optimized', 50, Mutation2(),
                                                    training, validation, testing, True)

semantic_learning_machine.create_initial_population()

semantic_learning_machine.mutate_champion()