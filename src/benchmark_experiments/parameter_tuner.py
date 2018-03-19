from algorithms.semantic_learning_machine.mutation_operators import Mutation2
from algorithms.semantic_learning_machine.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
from algorithms.simple_genetic_algorithm.selection_operator import SelectionOperatorTournament
from algorithms.simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from algorithms.simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from algorithms.simple_genetic_algorithm.neural_network.neural_network import create_network_from_topology
from itertools import product
from numpy import arange


_BASE_PARAMETERS = {
    'number_generations': 200,
    'population_size': 100
}

_SLM_FLS_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(_BASE_PARAMETERS.get('number_generations'))],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'layers': [1, 2, 3],
    'learning_step': [0.01],
    'max_connections': [1, 10, 50],
    'mutation_operator': [Mutation2()]
}

_SLM_OLS_PARAMETERS = {
    'stopping_criterion': [ErrorDeviationVariationCriterion(0.25)],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'layers': [1, 2, 3],
    'learning_step': ['optimized'],
    'max_connections': [1, 10, 50],
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
    'topology': [create_network_from_topology(topology) for topology in [[1], [2], [2, 2], [3, 3, 3], [5, 5, 5]]],
    'selection_operator': [SelectionOperatorTournament(5)],
    'mutation_operator': [MutationOperatorGaussian(0.01), MutationOperatorGaussian(0.1)],
    'crossover_operator': [CrossoverOperatorArithmetic()],
    'mutation_rate': [0.25, 0.5, 1],
    'crossover_rate': [0.01, 0.1, 0.25]
}

_SVM_PARAMETERS = {
    'C': [c / 10 for c in range(1, 11)],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'epsilon': [e / 10 for e in range(1, 11)],
    'degree': [d for d in range(1, 5)],
    'gamma': [g / 10 for g in range(1, 6)],
    'coef0': [co / 10 for co in range(1, 11)],
}

_MLP_PARAMETERS = {
    'hidden_layer_sizes': [(1), (2), (2, 2), (3, 3, 3), (5, 5, 5), (2, 2, 2, 2)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [10 ** -x for x in range(1, 7)],
    'learning_rate_init': [10 ** -x for x in range(1, 7)]
}

def _create_svc_configuration_list(list_dict):
    configuration_list = list()
    for kernel in _SVM_PARAMETERS.get('kernel'):
        if kernel == 'linear':
            keys = []
        if kernel == 'poly':
            keys = ['degree', 'gamma', 'coef0']
        if kernel == 'rbf':
            keys = ['gamma']
        if kernel == 'sigmoid':
            keys = ['gamma', 'coef0']
        keys.append('C')
        sub_dict = {k: _SVM_PARAMETERS[k] for k in keys if k in _SVM_PARAMETERS}
        sub_dict['kernel'] = [kernel]
        configuration_list.extend(_create_configuration_list(sub_dict))
    return configuration_list

def _create_svr_configuration_list(list_dict):
    configuration_list = list()
    for kernel in _SVM_PARAMETERS.get('kernel'):
        if kernel == 'linear':
            keys = []
        if kernel == 'poly':
            keys = ['degree', 'gamma', 'coef0']
        if kernel == 'rbf':
            keys = ['gamma']
        if kernel == 'sigmoid':
            keys = ['gamma', 'coef0']
        keys.append('C')
        keys.append('epsilon')
        sub_dict = {k: _SVM_PARAMETERS[k] for k in keys if k in _SVM_PARAMETERS}
        sub_dict['kernel'] = [kernel]
        configuration_list.extend(_create_configuration_list(sub_dict))
    return configuration_list

def _create_configuration_list(list_dict):
    return [{k:v for k, v in zip(list_dict.keys(), configuration)}
            for configuration in list(product(*[list_dict[key] for key in list_dict.keys()]))]

SLM_FLS_CONFIGURATIONS = _create_configuration_list(_SLM_FLS_PARAMETERS)
SLM_OLS_CONFIGURATIONS = _create_configuration_list(_SLM_OLS_PARAMETERS)
NEAT_CONFIGURATIONS = _create_configuration_list(_NEAT_PARAMETERS)
SGA_CONFIGURATIONS = _create_configuration_list(_SGA_PARAMETERS)
SVC_CONFIGURATIONS = _create_svc_configuration_list(_SVM_PARAMETERS)
SVR_CONFIGURATIONS = _create_svr_configuration_list(_SVM_PARAMETERS)
MLP_CONFIGURATIONS = _create_configuration_list(_MLP_PARAMETERS)

