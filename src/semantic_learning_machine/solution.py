from numpy import abs
from utils.calculations import root_mean_squared_error
from copy import copy


class Solution(object):
    """"""
    def __init__(self, ancestor, neural_network, target):
        self.neural_network = neural_network
        self.error = self.calculate_absolute_error(target)
        self.mean_error = self.calculate_mean_error()
        self.better_than_ancestor = self._better_than(ancestor)

    def _better_than(self, solution):
        return self.mean_error < solution.mean_error if solution else None

    def calculate_absolute_error(self, target):
        return self.neural_network.get_predictions() - target

    def calculate_mean_error(self):
        return root_mean_squared_error(abs(self.error))