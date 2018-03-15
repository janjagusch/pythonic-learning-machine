from copy import copy, deepcopy
from numpy.random import uniform
from simple_genetic_algorithm.solution import create_solution
from data.data_set import get_target_variable
from simple_genetic_algorithm.neural_network.connection import Connection
from simple_genetic_algorithm.neural_network.node import Sensor

class SimpleGeneticAlgorithm(object):
    """Class represents simple genetic algorithm.

    Algorithms breeds optimal weights for connections on fixed-topology neural network."""

    def __init__(self, stopping_criterion, population_size, topology,
                 selection_operator, mutation_operator, crossover_operator,
                 mutation_rate, crossover_rate, training_set):
        self.stopping_criterion = stopping_criterion
        self.population_size = population_size
        self.topology = deepcopy(topology)
        self.topology.add_sensors(training_set)
        self.selection_operator = selection_operator
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.training_set = training_set
        self.population = list()
        self.champion = None
        self.current_generation = 0

    def _create_solution(self, weights, crossover_ancestor=None, mutation_ancestor=None, better_than_crossover_ancestors=None):
        """Initializes solution for population."""
        # Creates copy of topology.
        neural_network = deepcopy(self.topology)
        # Assigns a random weight to each connection in neural network.
        for connection, weight in zip(neural_network.get_connections(), weights): connection.weight = weight
        # Calculate predictions.
        neural_network.calculate()
        # Creates solution.
        solution = create_solution(mutation_ancestor, crossover_ancestor, neural_network, get_target_variable(self.training_set).as_matrix(),
                                   better_than_crossover_ancestors)
        # Wipe semantics of solution.
        solution.neural_network.wipe_semantics()
        # Returns solution.
        return solution

    def _initialize_solution(self):
        """Initializes solution for population."""
        # Get random weights.
        weights= [uniform(-1, 1) for connection in self.topology.get_connections()]

        # Create new solution with weights.
        solution = self._create_solution(weights)

        # Return solution.
        return solution

    def _initialize_population(self):
        # Load sensors in topology.
        self.topology.load_sensors(self.training_set)
        # Create solution until population size is met.
        for i in range(self.population_size):
            solution = self._initialize_solution()
            # Add solution to population.
            self.population.append(solution)

    def _crossover_solution(self, ancestors):
        ancestor_weights = [ancestor.get_weights() for ancestor in ancestors]
        if self.crossover_rate > uniform(0, 1):
            offspring_weights = self.crossover_operator._crossover_weights(ancestor_weights)
        else:
            offspring_weights = ancestor_weights
        return [self._create_solution(weights, crossover_ancestor=ancestors) for weights in offspring_weights]

    def _crossover_population(self):
        new_population = list()
        for i in range(1, self.population_size, 2):
            new_population.extend(self._crossover_solution([self.population[i - 1], self.population[i]]))
        self.population = new_population

    def _mutate_solution(self, ancestor):
        ancestor_weights = ancestor.get_weights()
        if self.mutation_rate > uniform(0, 1):
            offspring_weights = self.mutation_operator._mutate_weights(ancestor_weights)
        else:
            offspring_weights = ancestor_weights
        return self._create_solution(offspring_weights, mutation_ancestor=ancestor,
                                     better_than_crossover_ancestors=ancestor.better_than_crossover_ancestors)

    def _mutate_population(self):
        self.population = [self._mutate_solution(solution) for solution in self.population]


        pass

    def _select_population(self):
        new_population = [self.selection_operator.select_solution(self.population) for i in range(self.population_size)]
        self.population = new_population

    def _evaluate_champion(self):
        for solution in self.population:
            # If there is no champion, set solution as champion.
            if not self.champion:
                self.champion = solution
            # If error is less than champion, set solution as champion.
            elif solution.mean_error < self.champion.mean_error:
                self.champion = solution

    def run(self):
        stopping_criterion = False
        while not stopping_criterion:
            if self.current_generation == 0:
                self._initialize_population()
            else:
                self._select_population()
                self._crossover_population()
                self._mutate_population()
            self._evaluate_champion()
            stopping_criterion = self.stopping_criterion.evaluate(self)
            print(self.champion.mean_error)
            self.current_generation += 1
