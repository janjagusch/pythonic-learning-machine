from algorithm.algorithm import EvolutionaryAlgorithm
from numpy import array, append
from algorithm.neat_python.create_configuration import create_configuration, write_configuration, get_configuration_path, remove_configuration
from neat import Config, DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, Population
from neat.nn.feed_forward import FeedForwardNetwork
from neat.six_util import iteritems, itervalues

class CompleteExtinctionException(Exception):
    pass

class Neat(EvolutionaryAlgorithm):
    """
    Wrapper class for Neuroevolution of Augmented Topologies (NEAT) algorithm: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
    Internally, calls the neat-python package: http://neat-python.readthedocs.io/en/latest/neat_overview.html

    Attributes:
        population_size = Number of individuals under study.
        compatibility_threshold = Mind 'Notes'.
        compatibility_disjoint_coefficient = Mind 'Notes'.
        compatibility_weight_coefficient = Mind 'Notes'.
        conn_add_prob = Mind 'Notes'.
        conn_delete_prob = Mind 'Notes'.
        node_add_prob = Mind 'Notes'.
        node_delete_prob = Mind 'Notes'.
        weight_mutate_power: Mind 'Notes'.
        weight_mutate_rate: Mind 'Notes'.

    Notes:
        More information about parameters: http://neat-python.readthedocs.io/en/latest/config_file.html
    """

    def __init__(self, population_size, stopping_criterion,
                 compatibility_threshold, compatibility_disjoint_coefficient, compatibility_weight_coefficient,
                 conn_add_prob, conn_delete_prob, node_add_prob, node_delete_prob,
                 weight_mutate_rate, weight_mutate_power):
        super().__init__(population_size, stopping_criterion)
        self.compatibility_threshold = compatibility_threshold
        self.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
        self.compatibility_weight_coefficient = compatibility_weight_coefficient
        self.conn_add_prob = conn_add_prob
        self.conn_delete_prob = conn_delete_prob
        self.node_add_prob = node_add_prob
        self.node_delete_prob = node_delete_prob
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_mutate_power = weight_mutate_power

    def _create_configuration_dict(self):
        """Creates configuration dict for generating configuration file."""
        return {
            'population_size': self.population_size,
            'compatibility_threshold': self.compatibility_threshold,
            'compatibility_disjoint_coefficient': self.compatibility_disjoint_coefficient,
            'compatibility_weight_coefficient': self.compatibility_weight_coefficient,
            'conn_add_prob': self.conn_add_prob,
            'conn_delete_prob': self.conn_delete_prob,
            'node_add_prob': self.node_add_prob,
            'node_delete_prob': self.node_delete_prob,
            'num_inputs': self.input_matrix.shape[1],
            'weight_mutate_rate': self.weight_mutate_rate,
            'weight_mutate_power': self.weight_mutate_power
        }

    def _write_configuration(self):
        """Writes configuration.ini"""
        configuration_dictionary = self._create_configuration_dict()
        configuration = create_configuration(configuration_dictionary)
        write_configuration(configuration)

    def _eval_genomes(self, genomes, configuration):
        """Evaluates genome based on inverse root mean squared error to target."""
        for genome_id, genome in genomes:
            prediction = self._predict_genome(genome)
            value = self.metric.evaluate(prediction, self.target_vector)
            if not self.metric.greater_is_better: value = 1 / value
            genome.fitness = value

    def _epoch(self):
        if self.current_generation == 0:
            self.population = Population(self.configuration)
        else:
            # Create the next generation from the current generation.
            self.population.population = self.population.reproduction.reproduce(self.population.config, self.population.species,
                                                          self.population.config.pop_size, self.population.generation)

            # Check for complete extinction.
            if not self.population.species.species:
                self.population.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.population.config.reset_on_extinction:
                    self.population.population = self.population.reproduction.create_new(self.population.config.genome_type,
                                                                   self.population.config.genome_config,
                                                                   self.population.config.pop_size)
                else:
                    raise CompleteExtinctionException()

        # Evaluate all genomes using the user-provided function.
        self._eval_genomes(list(iteritems(self.population.population)), self.population.config)
        # Gather and report statistics.
        best = None
        for g in itervalues(self.population.population):
            if best is None or g.fitness > best.fitness:
                best = g
        # Track the best genome ever seen.
        if self.population.best_genome is None or best.fitness > self.population.best_genome.fitness:
            self.population.best_genome = best
        self.champion = self.population.best_genome
        # Divide the new population into species.
        self.population.species.speciate(self.population.config, self.population.population, self.population.generation)
        return self.stopping_criterion.evaluate(self)

    def _get_champion_value(self):
        value = self.champion.fitness
        if not self.metric.greater_is_better: value = 1 / value
        return value

    def _predict_genome(self, genome):
        neural_network = FeedForwardNetwork.create(genome, self.configuration)
        predictions = array([])
        for data in self.input_matrix:
            predictions = append(predictions, float(neural_network.activate(data)[0]))
        return predictions

    def fit(self, input_matrix, target_vector, metric, verbose=False):
        self.input_matrix = input_matrix
        self.target_vector = target_vector
        self.metric = metric
        self._write_configuration()
        self.configuration = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, get_configuration_path())
        remove_configuration()
        self._run(verbose)
        self.input_matrix = None
        self.target_vector = None

    def predict(self, input_matrix):
        neural_network = FeedForwardNetwork.create(self.champion, self.configuration)
        predictions = array([])
        for data in input_matrix:
            predictions = append(predictions, float(neural_network.activate(data)[0]))
        return predictions
