from data.io_data_set import load_samples
from time import time
from src.benchmarker.parameter_tuner import SLM_CONFIGURATIONS, NEAT_CONFIGURATIONS
from semantic_learning_machine.semantic_learning_machine import SemanticLearningMachine
from copy import deepcopy
from utils.calculations import root_mean_squared_error
from data.data_set import get_target_variable
from neat_python.neat_python import Neat
from neat import Population

class Benchmarker(object):

    def __init__(self, data_set_name):
        self.samples = [load_samples(data_set_name, index) for index in range(30)]

    def run(self):
        for training, validation, testing in self.samples:
            slm_benchmark = _slm_benchmark(training, validation, testing)
            neat_benchmark = _neat_benchmark(training, validation, testing)

            # Do something with SLM.
            # Do something with NEAT.
            # Do something with SGA.


# SLM helper functions.
def _slm_benchmark(training, validation, testing):
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
    for slm_configuration in SLM_CONFIGURATIONS:
        # Create model from configuration.
        model = SemanticLearningMachine(training_set=training, **slm_configuration)
        # Compute network evolution.
        network_evolution = _slm_run(model)
        # Compute validation error.
        validation_error = _slm_calculate_validation_error(network_evolution, validation)
        # Add configuration and validation error to list.
        validation_error_list.append((slm_configuration, validation_error))
        # If validation error of configuration is less than current best, set as current best.
        if validation_error < best_validation_error:
            best_configuration = slm_configuration
            best_network_evolution = network_evolution
            best_validation_error = validation_error

    # For best configuration, retrieve training error.
    training_error_evolution = _slm_get_training_error(best_network_evolution)


    # For best configuration, calculate testing error.
    testing_error_evolution = _slm_calculate_testing_error(best_network_evolution, testing)

    return training_error_evolution, validation_error_list, testing_error_evolution

def _slm_run(slm):
    log = list()
    stopping_criterion = False
    while (not stopping_criterion):
        print(slm.current_generation)
        start_time = time()
        if slm.current_generation == 0:
            slm._initialize_population()
        else:
            slm._mutate_population()
        stopping_criterion = slm.stopping_criterion.evaluate(slm)
        slm._override_current_champion()
        slm._wipe_population()
        slm.current_generation += 1
        end_time = time()
        elapsed_time = end_time - start_time
        log.append((slm.current_champion, elapsed_time))
    return log

def _slm_get_networks(network_evolution):
    return [deepcopy(network[0].neural_network) for network in network_evolution]

def _slm_get_final_network(network_evolution):
    return deepcopy(network_evolution[-1][0].neural_network)

def _slm_get_solutions(network_evolution):
    return [network[0] for network in network_evolution]

def _slm_get_training_error(network_evolution):
    return [solution.mean_error for solution in _slm_get_solutions(network_evolution)]

def _slm_calculate_testing_error(network_evolution, testing_set):
    """"""
    # Testing error evolution.
    testing_error_evolution = list()

    # For each network in evolution, calculate testing error.
    for network in _slm_get_networks(network_evolution):
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

def _slm_calculate_validation_error(network_evolution, validation_set):
    # Retrieve final (last) network from evolution.
    final_neural_network = _slm_get_final_network(network_evolution)
    # Load validation data into network.
    final_neural_network.load_sensors(validation_set)
    # Calculate predictions.
    final_neural_network.calculate()
    # Calculate error.
    validation_error = root_mean_squared_error(get_target_variable(validation_set) -
                                               final_neural_network.get_predictions())
    # Return error.
    return validation_error

# NEAT helper functions.
def _neat_benchmark(training, validation, testing):
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
    for neat_configurations in NEAT_CONFIGURATIONS:
        # Create model from configuration.
        model = Neat(training_set=training, **neat_configurations)
        # Compute network evolution.
        network_evolution = _neat_run(model)
        # Compute validation error.
        validation_error = _slm_calculate_validation_error(network_evolution, validation)
        # Add configuration and validation error to list.
        validation_error_list.append((slm_configuration, validation_error))
        # If validation error of configuration is less than current best, set as current best.
        if validation_error < best_validation_error:
            best_configuration = slm_configuration
            best_network_evolution = network_evolution
            best_validation_error = validation_error

    # For best configuration, retrieve training error.
    training_error_evolution = _slm_get_training_error(best_network_evolution)


    # For best configuration, calculate testing error.
    testing_error_evolution = _slm_calculate_testing_error(best_network_evolution, testing)

    return training_error_evolution, validation_error_list, testing_error_evolution

def _neat_run(model):

    # Network evolution.
    log = list()

    population = Population(self.configuration)

    if self.config.no_fitness_termination and (n is None):
        raise RuntimeError("Cannot have no generational limit with no fitness termination")

    k = 0
    while n is None or k < n:
        k += 1

        self.reporters.start_generation(self.generation)

        # Evaluate all genomes using the user-provided function.
        fitness_function(list(iteritems(self.population)), self.config)

        # Gather and report statistics.
        best = None
        for g in itervalues(self.population):
            if best is None or g.fitness > best.fitness:
                best = g
        self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # Track the best genome ever seen.
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best

        if not self.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
            if fv >= self.config.fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, best)
                break

        # Create the next generation from the current generation.
        self.population = self.reproduction.reproduce(self.config, self.species,
                                                      self.config.pop_size, self.generation)

        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                               self.config.genome_config,
                                                               self.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        self.species.speciate(self.config, self.population, self.generation)

        self.reporters.end_generation(self.config, self.population, self.species)

        self.generation += 1

    if self.config.no_fitness_termination:
        self.reporters.found_solution(self.config, self.generation, self.best_genome)

    return self.best_genome




benchmarker = Benchmarker('r_concrete')

benchmarker.run()