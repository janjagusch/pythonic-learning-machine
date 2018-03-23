from copy import deepcopy
from numpy.random import uniform
from algorithm.simple_genetic_algorithm.solution import Solution
from algorithm.common.neural_network.neural_network import NeuralNetwork
from algorithm.common.algorithm import EvolutionaryAlgorithm

class SimpleGeneticAlgorithm(EvolutionaryAlgorithm):
    """Class represents simple genetic algorithm.

    Algorithms breeds optimal weights for connections on fixed-topology neural network."""

    def __init__(self, population_size, stopping_criterion,
                 topology, selection_operator, mutation_operator, crossover_operator,
                 mutation_rate, crossover_rate):
        super().__init__(population_size, stopping_criterion)
        self.topology = deepcopy(topology)
        self.selection_operator = selection_operator
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _create_neural_network(self, weights):
        # Creates copy of topology.
        neural_network = self._copy_topology()
        # Assigns a random weight to each connection in neural network.
        for connection, weight in zip(neural_network.get_connections(), weights): connection.weight = weight
        # Calculate predictions.
        neural_network.calculate()
        # Return neural network.
        return neural_network

    def _create_solution_from_network(self, neural_network, crossover_ancestors=None, mutation_ancestor=None, better_than_crossover_ancestors=None):
        prediction = neural_network.get_predictions()
        target = self.target_vector
        value = self._evaluate(prediction, target)
        solution = Solution(neural_network, value, None, None)
        solution.better_than_mutation_ancestor = self._is_better_solution(solution, mutation_ancestor)
        if not crossover_ancestors:
            solution.better_than_crossover_ancestors = better_than_crossover_ancestors
        else:
            solution.better_than_crossover_ancestors = all([self._is_better_solution(solution, crossover_ancestor)]
                                                           for crossover_ancestor in crossover_ancestors)
        return solution

    def _create_solution_from_weights(self, weights, crossover_ancestors=None, mutation_ancestor=None, better_than_crossover_ancestors=None):
        neural_network = self._create_neural_network(weights)
        solution = self._create_solution_from_network(neural_network, crossover_ancestors, mutation_ancestor, better_than_crossover_ancestors)
        return solution

    def _copy_topology(self):
        memodict = {}
        bias = self.topology.bias
        memodict[id(self.topology.bias)] = bias
        sensors = self.topology.sensors
        for sensor in sensors: memodict[id(sensor)] = sensor
        hidden_layers = deepcopy(self.topology.hidden_layers, memodict)
        output_neuron = deepcopy(self.topology.output_neuron, memodict)
        return NeuralNetwork(sensors, bias, hidden_layers, output_neuron)

    def _initialize_solution(self):
        """Initializes solution for population."""
        # Get random weights.
        weights= [uniform(-1, 1) for connection in self.topology.get_connections()]

        # Create new solution with weights.
        solution = self._create_solution_from_weights(weights)

        # Return solution.
        return solution

    def _initialize_population(self):
        # Create solution until population size is met.
        for i in range(self.population_size):
            solution = self._initialize_solution()
            # Add solution to population.
            self.population.append(solution)

    def _crossover_solution(self, ancestors):
        ancestor_weights = [ancestor.neural_network.get_weights() for ancestor in ancestors]
        if self.crossover_rate > uniform(0, 1):
            offspring_weights = self.crossover_operator._crossover_weights(ancestor_weights)
        else:
            offspring_weights = ancestor_weights
        return [self._create_solution_from_weights(weights, crossover_ancestors=ancestors) for weights in offspring_weights]

    def _crossover_population(self):
        new_population = list()
        for i in range(1, self.population_size, 2):
            new_population.extend(self._crossover_solution([self.population[i - 1], self.population[i]]))
        self.population = new_population

    def _mutate_solution(self, ancestor):
        ancestor_weights = ancestor.neural_network.get_weights()
        if self.mutation_rate > uniform(0, 1):
            offspring_weights = self.mutation_operator._mutate_weights(ancestor_weights)
        else:
            offspring_weights = ancestor_weights
        return self._create_solution_from_weights(offspring_weights, mutation_ancestor=ancestor,
                                                  better_than_crossover_ancestors=ancestor.better_than_crossover_ancestors)

    def _mutate_population(self):
        self.population = [self._mutate_solution(solution) for solution in self.population]

    def _select_population(self):
        new_population = [self.selection_operator.select_solution(self.population, self.metric) for i in range(self.population_size)]
        self.population = new_population

    def _evaluate_champion(self):
        for solution in self.population:
            # If there is no champion, set solution as champion.
            if not self.champion:
                self.champion = solution
            # If error is less than champion, set solution as champion.
            elif self._is_better_solution(solution, self.champion):
                self.champion = solution

    def _epoch(self):
        if self.current_generation == 0:
            self.topology.add_sensors(self.input_matrix)
            self._initialize_population()
        else:
            self._select_population()
            self._crossover_population()
            self._mutate_population()
        self._evaluate_champion()
        return self.stopping_criterion.evaluate(self)

    def fit(self, input_matrix, target_vector, metric, verbose=False):
        super().fit(input_matrix, target_vector, metric, verbose)
        self.champion.neural_network = deepcopy(self.champion.neural_network)
        self.population = None

    def predict(self, input_matrix):
        neural_network = self.champion.neural_network
        neural_network.load_sensors(input_matrix)
        neural_network.calculate()
        return neural_network.get_predictions()