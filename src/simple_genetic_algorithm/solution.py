from numpy import array
from utils.calculations import root_mean_squared_error
from copy import copy


class Solution(object):
    """"""
    def __init__(self, neural_network, error, mean_error,
                 better_than_mutation_ancestor, better_than_crossover_ancestors):
        self.neural_network = neural_network
        self.error = error
        self.mean_error = mean_error
        self.better_than_mutation_ancestor = better_than_mutation_ancestor
        self.better_than_crossover_ancestors = better_than_crossover_ancestors

    def _better_than(self, solution):
        return self.mean_error < solution.mean_error

    def _calculate_error(self, target):
        return self.neural_network.get_predictions() - target

    def _better_than_ancestor(self):
        return self.better_than_mutation_ancestor and self.better_than_crossover_ancestors

    def calculate_mean_error(self):
        return root_mean_squared_error(self.error)

    def get_weights(self):
        connections = self.neural_network.get_connections()
        return array([connection.weight for connection in connections])

def create_solution(mutation_ancestor, crossover_ancestors, neural_network, target, better_than_crossover_ancestors):
    solution = Solution(neural_network, None, None, None, None)
    solution.error = solution._calculate_error(target)
    solution.mean_error = solution.calculate_mean_error()
    if mutation_ancestor:
        solution.better_than_mutation_ancestor = solution._better_than(mutation_ancestor)
    if crossover_ancestors:
        solution.better_than_crossover_ancestors = all(solution._better_than(ancestor) for ancestor in crossover_ancestors)
    if better_than_crossover_ancestors is not None:
        solution.better_than_crossover_ancestors = better_than_crossover_ancestors
    return solution

