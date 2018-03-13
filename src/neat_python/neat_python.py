from data.data_set import get_input_variables, get_target_variable
from utils.calculations import root_mean_squared_error
from numpy import array, append
from neat_python.create_configuration import create_configuration, write_configuration, get_configuration_path, remove_configuration
from neat import Config, DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, Population, \
    StdOutReporter, StatisticsReporter
from neat.nn.feed_forward import FeedForwardNetwork


class Neat(object):

    def __init__(self, training_set, number_generations, population_size,
                 compatibility_threshold, compatibility_disjoint_coefficient, compatibility_weight_coefficient,
                 conn_add_prob, conn_delete_prob, node_add_prob, node_delete_prob):
        self.training_set = training_set
        self.number_generations = number_generations
        self.population_size = population_size
        self.compatibility_threshold = compatibility_threshold
        self.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
        self.compatibility_weight_coefficient = compatibility_weight_coefficient
        self.conn_add_prob = conn_add_prob
        self.conn_delete_prob = conn_delete_prob
        self.node_add_prob = node_add_prob
        self.node_delete_prob = node_delete_prob
        self._write_configuration()
        self.configuration = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation,
                                    get_configuration_path())
        remove_configuration()


    def _create_configuration_dict(self):
        return {
            'population_size': self.population_size,
            'compatibility_threshold': self.compatibility_threshold,
            'compatibility_disjoint_coefficient': self.compatibility_disjoint_coefficient,
            'compatibility_weight_coefficient': self.compatibility_weight_coefficient,
            'conn_add_prob': self.conn_add_prob,
            'conn_delete_prob': self.conn_delete_prob,
            'node_add_prob': self.node_add_prob,
            'node_delete_prob': self.node_delete_prob,
            'num_inputs': get_input_variables(self.training_set).shape[1]
        }

    def _write_configuration(self):
        configuration_dictionary = self._create_configuration_dict()
        configuration = create_configuration(configuration_dictionary)
        write_configuration(configuration)


    def run(self):
        pass

    def _eval_genomes(self, genomes, configuration):
        for genome_id, genome in genomes:
            neural_network = FeedForwardNetwork.create(genome, self.configuration)
            predictions = array([])
            for row in get_input_variables(self.training_set).iterrows():
                index, data = row
                predictions = append(predictions, float(neural_network.activate(data)[0]))
            error = predictions - get_target_variable(self.training_set).as_matrix()
            mean_error = root_mean_squared_error(error)
            genome.fitness = 1 / mean_error

    def run(self):
        population = Population(self.configuration)

        population.add_reporter(StdOutReporter(True))
        population.add_reporter(StatisticsReporter())

        population.run(self._eval_genomes, self.number_generations)

