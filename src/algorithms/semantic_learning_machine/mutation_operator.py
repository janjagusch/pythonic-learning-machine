from random import randint, sample
from algorithms.common.neural_network.neural_network import create_neuron

class Mutation(object):
    """
    Class represents mutation operator for semantic learning machine.
    """

    def mutate_network(self, algorithm):
        pass

    def _create_final_hidden_neuron(self, bias):
        """Creates the final hidden neuron, which must have a hyperbolic tangent activation function."""
        return create_neuron('tanh', bias)

class Mutation1(Mutation):
    """Adds one neuron to the last hidden layer."""

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        # Creates empty layers
        hidden_layers = [[] for i in len(neural_network.hidden_layers) - 1]
        # Adds one neuron to the last hidden layer.
        hidden_layers.append([self._create_final_hidden_neuron(bias)])
        return hidden_layers

class Mutation2(Mutation):
    """Adds one neuron the each hidden layer."""

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        hidden_layers = [[create_neuron(activation_function=None, bias=bias)]
                         for i in range(len(neural_network.hidden_layers) - 1)]
        hidden_layers.append([self._create_final_hidden_neuron(bias)])
        return hidden_layers

class Mutation3(Mutation):
    """Adds an equal, random number of neurons to each hidden layer."""

    def __init__(self, max_neurons):
        self.max_neurons = max_neurons

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        neurons = randint(1, self.max_neurons)
        hidden_layers = [[create_neuron() for i in range(neurons)] for j in range(len(neural_network.hidden_layers) - 1)]
        hidden_layers.append([self._create_final_hidden_neuron(bias)])
        return hidden_layers

class Mutation4(Mutation):
    """Adds a distinct, random number of neurons to each hidden layer."""

    def __init__(self, max_neurons):
        self.max_neurons = max_neurons

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        number_layers = len(neural_network.hidden_layers)
        neurons = self.max_neurons if self.max_neurons >= number_layers else number_layers
        neurons = sample(range(1, neurons), number_layers - 1)
        hidden_layers = [[create_neuron() for i in range(neuron)] for neuron in neurons]
        hidden_layers.append([self._create_final_hidden_neuron(bias)])
        return hidden_layers
