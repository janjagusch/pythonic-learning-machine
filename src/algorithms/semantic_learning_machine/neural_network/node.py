from numpy import sum, array
from algorithms.semantic_learning_machine.neural_network.activation_function import calculate_output
from copy import copy, deepcopy


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

    def __deepcopy__(self, memodict={}):
        sensor = Sensor(array([]))
        memodict[id(self)] = sensor
        return sensor

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

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return Neuron(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = array([])
        neuron = Neuron(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron


    def _calculate_weighted_input(self):
        """Calculates weighted input from input connections."""
        return sum([connection.from_node.semantics * connection.weight for connection in self.input_connections], axis=0)

    def _calculate_output(self, weighted_input):
        """Calculates semantics, based on weighted input."""
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        """Calculates weighted input, then calculates semantics."""
        weighted_input = self._calculate_weighted_input()
        self.semantics = self._calculate_output(weighted_input)