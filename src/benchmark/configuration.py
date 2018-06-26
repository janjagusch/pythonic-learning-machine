from algorithms.common.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
from algorithms.common.neural_network.neural_network import create_network_from_topology
from algorithms.semantic_learning_machine.mutation_operator import Mutation2, Mutation3, Mutation4
from algorithms.simple_genetic_algorithm.selection_operator import SelectionOperatorTournament
from algorithms.simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from algorithms.simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from itertools import product
from numpy import mean


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
    'stopping_criterion': [ErrorDeviationVariationCriterion(0.25), ErrorDeviationVariationCriterion(0.5)],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'learning_step': ['optimized'],
    'max_connections': [1, 10, 50, 100],
    'mutation_operator': [Mutation2()]
}

_NEAT_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(_BASE_PARAMETERS.get(('number_generations')))],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'compatibility_threshold': [3, 4],
    'compatibility_disjoint_coefficient': [1],
    'compatibility_weight_coefficient': [1],
    'conn_add_prob': [0.1, 0.25],
    'conn_delete_prob': [0.1],
    'node_add_prob': [0.1, 0.25],
    'node_delete_prob': [0.1],
    'weight_mutate_rate': [0.25],
    'weight_mutate_power': [0.25]
}

_SGA_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(_BASE_PARAMETERS.get('number_generations'))],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'topology': [create_network_from_topology(topology) for topology in [[1], [2], [2, 2], [3, 3, 3], [5, 5, 5]]],
    'selection_operator': [SelectionOperatorTournament(5)],
    'mutation_operator': [MutationOperatorGaussian(0.01), MutationOperatorGaussian(0.1)],
    'crossover_operator': [CrossoverOperatorArithmetic()],
    'mutation_rate': [0.25, 0.5],
    'crossover_rate': [0.01, 0.1]
}

_SVM_PARAMETERS = {
    'C': [c / 10 for c in range(1, 11)],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'epsilon': [e / 10 for e in range(1, 6)],
    'degree': [d for d in range(1, 5)],
    'gamma': [g / 10 for g in range(1, 6)],
    'coef0': [co / 10 for co in range(1, 6)],
    'probability': [True]
}

_MLP_PARAMETERS = {
    'hidden_layer_sizes': [(1), (2), (2, 2), (3, 3, 3), (5, 5, 5), (2, 2, 2, 2)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [10 ** -x for x in range(1, 7)],
    'learning_rate_init': [10 ** -x for x in range(1, 7)]
}

_RF_PARAMETERS = {
    'n_estimators': [25],
    'max_depth': [1, 2, 5, None],
    'min_samples_split': [0.01, 0.02, 0.05]
}

def _create_base_learner(algorithm, configurations):
    return [algorithm(**configuration) for configuration in configurations]

def _create_svc_configuration_list(list_dict):
    configuration_list = list()
    for kernel in list_dict.get('kernel'):
        if kernel == 'linear':
            keys = []
        if kernel == 'poly':
            keys = ['degree', 'gamma', 'coef0']
        if kernel == 'rbf':
            keys = ['gamma']
        if kernel == 'sigmoid':
            keys = ['gamma', 'coef0']
        keys.append('C')
        keys.append('probability')
        sub_dict = {k: list_dict[k] for k in keys if k in list_dict}
        sub_dict['kernel'] = [kernel]
        configuration_list.extend(_create_configuration_list(sub_dict))
    return configuration_list

def _create_svr_configuration_list(list_dict):
    configuration_list = list()
    for kernel in list_dict.get('kernel'):
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
        sub_dict = {k: list_dict[k] for k in keys if k in list_dict}
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
RF_CONFIGURATIONS = _create_configuration_list(_RF_PARAMETERS)

_ENSEMBLE_PARAMETERS = {
    'base_learner': _create_base_learner(SemanticLearningMachine, SLM_OLS_CONFIGURATIONS),
    'number_learners': [25],
    'meta_learner': [mean]
}

ENSEMBLE_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_PARAMETERS)
