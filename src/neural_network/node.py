from numpy import sum
from neural_network.activation_function import calculate_output


class Node(object):
    """
    Class represents abstract node in neural network.

    Attributes:
        semantics: Semantic vector
    """

    def __init__(self, semantics):
        self.semantics = semantics


class Sensor(Node):
    """
    Class represents input sensor in neural network.
    """


class Neuron(Node):
    """
    Class represents neuron in neural network.

    Attributes:
        input_connections = Set of input connections
        activation_function = String for activation function id
    """

    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics)
        self.input_connections = input_connections
        self.activation_function = activation_function

    def _calculate_weighted_input(self):
        return sum([connection.from_node.semantics * connection.weight for connection in self.input_connections], axis=0)

    def _calculate_output(self, weighted_input):
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        weighted_input = self._calculate_weighted_input()
        self.semantics = self._calculate_output(weighted_input)
