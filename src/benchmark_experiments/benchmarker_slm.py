from algorithms.semantic_learning_machine.semantic_learning_machine import SemanticLearningMachine
from datetime import datetime
from copy import deepcopy
from utils.calculations import root_mean_squared_error
from data.data_set import get_target_variable
from timeout_decorator import timeout

def slm_benchmark(training, validation, testing, configurations):
    """"""
    # Validation error for all configurations.
    validation_error_list = list()
    # Best configuration (lowest validation error).
    best_configuration = None
    # Best evolution of network.
    best_network_evolution = None
    # Best (lowest) validation error.
    best_validation_error = float('Inf')

    i = 1

    # Iterate through all configurations.
    for configuration in configurations:
        print(i)
        # Create model from configuration.
        model = SemanticLearningMachine(training_set=training, **configuration)
        # Compute network evolution.
        network_evolution = _run(model)
        # Compute validation error.
        validation_error = _calculate_validation_error(network_evolution, validation)
        # Add configuration and validation error to list.
        validation_error_list.append((configuration, validation_error))
        # If validation error of configuration is less than current best, set as current best.
        if validation_error < best_validation_error:
            best_configuration = configuration
            best_network_evolution = network_evolution
            best_validation_error = validation_error
        i += 1

    print('Done with validation...')

    # Retrieve topology from best evolution log.
    topology = _get_topology(best_network_evolution)

    # For best configuration, retrieve training error.
    training_error_evolution = _get_training_error(best_network_evolution)

    # For best configuration, calculate testing error.
    testing_error_evolution = _calculate_testing_error(best_network_evolution, testing)

    processing_time = _get_processing_time(best_network_evolution)

    return training_error_evolution, validation_error_list, testing_error_evolution, topology, processing_time

def _run(slm):
    log = list()
    stopping_criterion = False
    while (not stopping_criterion):
        start_time = datetime.now()
        if slm.current_generation == 0:
            slm._initialize_population()
        else:
            slm._mutate_population()
        stopping_criterion = slm.stopping_criterion.evaluate(slm)
        slm._override_current_champion()
        slm._wipe_population()
        slm.current_generation += 1
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        elapsed_time = elapsed_time.total_seconds()
        log.append((slm.current_champion, elapsed_time))
    return log

def _get_networks(network_evolution):
    return [deepcopy(network[0].neural_network) for network in network_evolution]

def _get_final_network(network_evolution):
    return deepcopy(network_evolution[-1][0].neural_network)

def _get_solutions(network_evolution):
    return [network[0] for network in network_evolution]

def _get_training_error(network_evolution):
    return [solution.mean_error for solution in _get_solutions(network_evolution)]

def _calculate_testing_error(network_evolution, testing_set):
    """"""
    # Testing error evolution.
    testing_error_evolution = list()

    # For each network in evolution, calculate testing error.
    for network in _get_networks(network_evolution):
        # Create deepcopy from network.
        network_copy = deepcopy(network)
        # Load testing data into network.
        network_copy.load_sensors(testing_set)
        # Calculate predictions.
        network_copy.calculate()
        # Calculate error.
        testing_error = root_mean_squared_error(get_target_variable(testing_set) - network_copy.get_predictions())
        # Append error to list.
        testing_error_evolution.append(testing_error)

    # Return testing error evolution.
    return testing_error_evolution

def _calculate_validation_error(network_evolution, validation_set):
    # Retrieve final (last) network from evolution.
    final_neural_network = _get_final_network(network_evolution)
    # Load validation data into network.
    final_neural_network.load_sensors(validation_set)
    # Calculate predictions.
    final_neural_network.calculate()
    # Calculate error.
    validation_error = root_mean_squared_error(get_target_variable(validation_set) -
                                               final_neural_network.get_predictions())
    # Return error.
    return validation_error

def _get_topology(network_evolution):
    networks = _get_networks(network_evolution)
    return [network.get_topology() for network in networks]

def _get_processing_time(network_evolution):
    return [network[1] for network in network_evolution]