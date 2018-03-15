from semantic_learning_machine.mutation_operators import Mutation1, Mutation2, Mutation3, Mutation4
from semantic_learning_machine.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
from simple_genetic_algorithm.selection_operator import SelectionOperatorTournament
from simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from simple_genetic_algorithm.neural_network.neural_network import create_network_from_topology
from itertools import product


_BASE_PARAMETERS = {
    'number_generations': 10,
    'population_size': 10
}

_SLM_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(_BASE_PARAMETERS.get('number_generations'))],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'layers': [1, 2],
    'learning_step': [0.01, 'optimized'],
    'max_connections': [1, 10],
    'mutation_operator': [Mutation2()]
}

_NEAT_PARAMETERS = {
    'number_generations': [_BASE_PARAMETERS.get('number_generations')],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'compatibility_threshold': [1],
    'compatibility_disjoint_coefficient': [1],
    'compatibility_weight_coefficient': [1, 2],
    'conn_add_prob': [0.1],
    'conn_delete_prob': [0.1],
    'node_add_prob': [0.1],
    'node_delete_prob': [0.1]
}

_SGA_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(_BASE_PARAMETERS.get('number_generations'))],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'topology': [create_network_from_topology(topology) for topology in [[1], [2], [2, 2], [3, 3, 3], [5, 5, 5], [2, 2, 2, 2]]],
    'selection_operator': [SelectionOperatorTournament(5)],
    'mutation_operator': [MutationOperatorGaussian(0.01)],
    'crossover_operator': [CrossoverOperatorArithmetic()],
    'mutation_rate': [0.01, 0.1, 0.25, 0.5, 1],
    'crossover_rate': [0.01, 0.1, 0.25, 0.5, 1]
}


def _create_configuration_list(list_dict):
    return [{k:v for k, v in zip(list_dict.keys(), configuration)}
            for configuration in list(product(*[list_dict[key] for key in list_dict.keys()]))]

SLM_CONFIGURATIONS = _create_configuration_list(_SLM_PARAMETERS)
NEAT_CONFIGURATIONS = _create_configuration_list(_NEAT_PARAMETERS)
SGA_CONFIGURATION = _create_configuration_list(_SGA_PARAMETERS)
