from copy import copy


class Solution(object):
    """"""
    def __init__(self, neural_network, value, better_than_ancestor):
        self.neural_network = neural_network
        self.value = value
        self.better_than_ancestor = better_than_ancestor
        self.predictions = copy(neural_network.get_predictions())
