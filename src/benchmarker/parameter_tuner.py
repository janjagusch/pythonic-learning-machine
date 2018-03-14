from semantic_learning_machine.mutation_operators import Mutation1, Mutation2, Mutation3, Mutation4
from semantic_learning_machine.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
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

def _create_configuration_list(list_dict):
    return [{k:v for k, v in zip(list_dict.keys(), configuration)}
            for configuration in list(product(*[list_dict[key] for key in list_dict.keys()]))]

SLM_CONFIGURATIONS = _create_configuration_list(_SLM_PARAMETERS)
NEAT_CONFIGURATIONS = _create_configuration_list(_NEAT_PARAMETERS)
