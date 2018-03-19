from numpy.random import normal
from numpy import array

class MutationOperator(object):

    def _mutate_weights(self, weights):
        pass


class MutationOperatorGaussian(MutationOperator):
    """Applies a Gaussian perturbation with defined standard deviation to every weight."""

    def __init__(self, standard_deviation):
        self.standard_deviation = standard_deviation

    def _mutate_weights(self, weights):
        """Applies perturbation to numpy array of weights."""
        return weights + normal(loc=0, scale=self.standard_deviation, size=weights.shape[0])
