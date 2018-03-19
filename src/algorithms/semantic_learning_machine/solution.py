from numpy import abs
from utils.calculations import root_mean_squared_error
from copy import copy


class Solution(object):
    """"""
    def __init__(self, neural_network, error, mean_error, better_than_ancestor):
        self.neural_network = neural_network
        self.error = error
        self.mean_error = mean_error
        self.better_than_ancestor = better_than_ancestor

    def __copy__(self):
        copy_neural_network = copy(self.neural_network)
        copy_error = copy(self.error)
        copy_mean_error = self.mean_error
        copy_better_than_ancestor = self.better_than_ancestor
        return Solution(copy_neural_network, copy_error, copy_mean_error, copy_better_than_ancestor)

    def _better_than(self, solution):
        return self.mean_error < solution.mean_error if solution else None

    def _calculate_error(self, target):
        return self.neural_network.get_predictions() - target

    def calculate_mean_error(self):
        return root_mean_squared_error(abs(self.error))

def create_solution(ancestor, neural_network, target):
    solution = Solution(neural_network, None, None, None)
    solution.error = solution._calculate_error(target)
    solution.mean_error = solution.calculate_mean_error()
    solution.better_than_ancestor = solution._better_than(ancestor)
    return solution
