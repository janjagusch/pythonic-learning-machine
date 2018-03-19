from benchmark_experiments.parameter_tuner import NEAT_CONFIGURATIONS
from algorithms.neat_python.neat_python import Neat
from neat import Population, CompleteExtinctionException
from neat.six_util import iteritems, itervalues
from neat.nn import FeedForwardNetwork
from datetime import datetime
from numpy import array, append
from data.data_set import get_input_variables, get_target_variable
from utils.calculations import root_mean_squared_error


# NEAT helper functions.
def neat_benchmark(training, validation, testing):
    """"""
    # Validation error for all configurations.
    validation_error_list = list()
    # Best configuration (lowest validation error).
    best_configuration = None
    # Best evolution of network.
    best_network_evolution = None
    # Best (lowest) validation error.
    best_validation_error = float('Inf')
    # Best model.
    best_model = None

    # Iterate through all configurations.
    for configuration in NEAT_CONFIGURATIONS:
        # Create model from configuration.
        model = Neat(training_set=training, **configuration)
        # Compute network evolution.
        evolution_log = _run(model)
        # Compute validation error.
        validation_error = _calculate_validation_error(evolution_log, model.configuration, validation)
        # Add configuration and validation error to list.
        validation_error_list.append((configuration, validation_error))
        # If validation error of configuration is less than current best, set as current best.
        if validation_error < best_validation_error:
            best_configuration = configuration
            best_network_evolution = evolution_log
            best_validation_error = validation_error
            best_model = model

    # For best configuration, retrieve training error.
    training_error_evolution = _calculate_training_error(best_network_evolution)

    # For best configuration, calculate testing error.
    testing_error_evolution = _calculate_testing_error(best_network_evolution, best_model.configuration, testing)

    processing_time = _get_processing_time(best_network_evolution)

    return {
        'training_error': training_error_evolution,
        'validation_error': validation_error_list,
        'testing_error': testing_error_evolution,
        'topology': None,
        'processing_time': processing_time
    }

def _run(model):

    # Network evolution.
    log = list()

    population = Population(model.configuration)
    n = model.number_generations

    if population.config.no_fitness_termination and (n is None):
        raise RuntimeError("Cannot have no generational limit with no fitness termination")

    fitness_function = model._eval_genomes

    k = 0
    while n is None or k < n:
        start_time = datetime.now()
        k += 1

        # self.reporters.start_generation(self.generation)

        # Evaluate all genomes using the user-provided function.
        fitness_function(list(iteritems(population.population)), population.config)

        # Gather and report statistics.
        best = None
        for g in itervalues(population.population):
            if best is None or g.fitness > best.fitness:
                best = g
        # self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # Track the best genome ever seen.
        if population.best_genome is None or best.fitness > population.best_genome.fitness:
            population.best_genome = best

        # if not self.config.no_fitness_termination:
        #     End if the fitness threshold is reached.
        #     fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
        #     if fv >= self.config.fitness_threshold:
        #         self.reporters.found_solution(self.config, self.generation, best)
        #         break

        # Create the next generation from the current generation.
        population.population = population.reproduction.reproduce(population.config, population.species,
                                                      population.config.pop_size, population.generation)

        # Check for complete extinction.
        if not population.species.species:
        #     self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if population.config.reset_on_extinction:
                population.population = population.reproduction.create_new(population.config.genome_type,
                                                               population.config.genome_config,
                                                               population.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        population.species.speciate(population.config, population.population, population.generation)

        # self.reporters.end_generation(self.config, self.population, self.species)

        population.generation += 1

        # Track end time.
        end_time = datetime.now()

        # Calculate elapsed time.
        elapsed_time = end_time - start_time
        elapsed_time = elapsed_time.total_seconds()

        # Add best genome ever seen and elapsed time to log
        log.append((population.best_genome, elapsed_time))

    # if self.config.no_fitness_termination:
    #     self.reporters.found_solution(self.config, self.generation, self.best_genome)

    return log

    # return self.best_genome

def _get_neural_network(genome, configuration):
    return FeedForwardNetwork.create(genome, configuration)

def _get_predictions(neural_network, data_set):
    prediction_array = array([])
    for row in get_input_variables(data_set).iterrows():
        index, data = row
        prediction = float(neural_network.activate(data)[0])
        prediction_array = append(prediction_array, prediction)
    return prediction_array

def _get_final_genome(evolution_log):
    return evolution_log[-1][0]

def _get_genomes(evolution_log):
    return [item[0] for item in evolution_log]

def _get_processing_time(evolution_log):
    return [item[1] for item in evolution_log]

def _calculate_error(genome, configuration, data_set):
    neural_network = _get_neural_network(genome, configuration)
    predictions = _get_predictions(neural_network, data_set)
    error = root_mean_squared_error(get_target_variable(data_set).as_matrix() - predictions)
    return error

def _calculate_validation_error(evolution_log, configuration, data_set):
    """"""
    # The final genome for evolution log.
    genome = _get_final_genome(evolution_log)
    return _calculate_error(genome, configuration, data_set)

def _calculate_training_error(evolution_log):
    genomes = _get_genomes(evolution_log)
    error = list()
    for genome in genomes:
        error.append(1 / genome.fitness)
    return error

def _calculate_testing_error(evolution_log, configuration, data_set):
    genomes = _get_genomes(evolution_log)
    error = list()
    for genome in genomes:
        error.append(_calculate_error(genome, configuration, data_set))
    return error

def _get_topology(neural_network):
    pass