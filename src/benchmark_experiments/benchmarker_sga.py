from algorithms.simple_genetic_algorithm.algorithm import SimpleGeneticAlgorithm
from benchmark_experiments.parameter_tuner import SGA_CONFIGURATIONS
from datetime import datetime
from data.data_set import get_target_variable
from utils.calculations import root_mean_squared_error
from copy import deepcopy
from utils.format import print_progress


def sga_benchmark(training, validation, testing):
    """"""
    # Validation error for all configurations.
    validation_error_list = list()
    # Best configuration (lowest validation error).
    best_configuration = None
    # Best evolution of network.
    best_network_evolution = None
    # Best (lowest) validation error.
    best_validation_error = float('Inf')

    # Iterate through all configurations.
    for sga_configuration in SGA_CONFIGURATIONS:
        # Create model from configuration.
        model = SimpleGeneticAlgorithm(training_set=training, **sga_configuration)
        # Compute network evolution.
        network_evolution = _run(model)
        # Compute validation error.
        validation_error = _calculate_validation_error(network_evolution, validation)
        # Add configuration and validation error to list.
        validation_error_list.append((sga_configuration, validation_error))
        # If validation error of configuration is less than current best, set as current best.
        if validation_error < best_validation_error:
            best_configuration = sga_configuration
            best_network_evolution = network_evolution
            best_validation_error = validation_error

    # Retrieve topology from best evolution log.
    best_topology = _get_topology(best_network_evolution)
    #
    # For best configuration, retrieve training error.
    training_error_evolution = _get_training_error(best_network_evolution)
    #
    # For best configuration, calculate testing error.
    testing_error_evolution = _calculate_testing_error(best_network_evolution, testing)

    return training_error_evolution, validation_error_list, testing_error_evolution, best_topology

def _run(model):
    log = list()
    stopping_criterion = False
    while not stopping_criterion:
        start_time = datetime.now()
        if model.current_generation == 0:
            model._initialize_population()
        else:
            model._select_population()
            model._crossover_population()
            model._mutate_population()
        model._evaluate_champion()
        stopping_criterion = model.stopping_criterion.evaluate(model)
        model.current_generation += 1
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        elapsed_time = elapsed_time.total_seconds()
        log.append((model.champion, elapsed_time))
    return log

def _calculate_error(neural_network, data_set):
    predictions = _calculate_predictions(neural_network, data_set)
    error = root_mean_squared_error(get_target_variable(data_set).as_matrix() - predictions)
    return error

def _calculate_predictions(neural_network, data_set):
    neural_network.load_sensors(data_set)
    neural_network.calculate()
    return neural_network.get_predictions()

def _get_solutions(network_evolution):
    return [item[0] for item in network_evolution]

def _get_final_neural_network(network_evolution):
    return deepcopy(network_evolution[-1][0].neural_network)

def _get_networks(network_evolution):
    return [deepcopy(solution.neural_network) for solution in _get_solutions(network_evolution)]

def _calculate_validation_error(network_evolution, validation_set):
    neural_network = _get_final_neural_network(network_evolution)
    return _calculate_error(neural_network, validation_set)

def _get_topology(network_evolution):
    networks = _get_networks(network_evolution)
    return [network.get_topology() for network in networks]

def _get_training_error(network_evolution):
    return [solution.mean_error for solution in _get_solutions(network_evolution)]

def _calculate_testing_error(network_evolution, testing_set):
    neural_networks = _get_networks(network_evolution)
    return [_calculate_error(neural_network, testing_set) for neural_network in neural_networks]
