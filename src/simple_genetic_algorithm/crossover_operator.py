from numpy.random import uniform
from numpy import array
from copy import copy
from simple_genetic_algorithm.solution import create_solution

class CrossoverOperator(object):

    def _crossover_weights(self, ancestor_weights):
        pass


class CrossoverOperatorArithmetic(CrossoverOperator):
    """Calculates weighted Arithmetic average between two weights."""

    def _crossover_weights(self, ancestor_weights):
        """Calculates weighted arithmetic average between two numpy arrays of weights."""
        # Get random weight for weighted average.
        crossover_factor = uniform(0, 1)
        # Create offspring weight list.
        offspring = list()
        # Append offspring to list.
        offspring.append(crossover_factor * ancestor_weights[0] + (1 - crossover_factor) * ancestor_weights[1])
        offspring.append((1 - crossover_factor) * ancestor_weights[0] + crossover_factor * ancestor_weights[1])
        # Return offspring weights.
        return offspring
