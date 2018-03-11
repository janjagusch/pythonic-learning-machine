from neural_network.node import create_neuron
from random import randint, sample

class Mutation(object):

    def mutate_network(self, neural_network):
        pass

    def _create_final_hidden_neuron(self):
        return create_neuron('tanh')

class Mutation1(Mutation):
    """Adds one neuron to the last hidden layer."""

    def mutate_network(self, neural_network):
        hidden_layers = [[] for i in len(neural_network.hidden_layers) - 1]
        hidden_layers.append([self._create_final_hidden_neuron()])
        return hidden_layers

class Mutation2(Mutation):
    """Adds one neuron the each hidden layer."""

    def mutate_network(self, neural_network):
        hidden_layers = [[create_neuron()] for i in len(neural_network) - 1]
        hidden_layers.append([self._create_final_hidden_neuron()])
        return hidden_layers

class Mutation3(Mutation):
    """Adds an equal, random number of neurons to each hidden layer."""

    def __init__(self, max_neurons):
        self.max_neurons = max_neurons

    def mutate_network(self, neural_network):
        neurons = randint(1, self.max_neurons)
        hidden_layers = [[create_neuron() for i in range(neurons)] for j in len(neural_network) - 1]
        hidden_layers.append([self._create_final_hidden_neuron()])
        return hidden_layers

class Mutation4(Mutation):
    """Adds a distinct, random number of neurons to each hidden layer."""

    def __init__(self, max_neurons):
        self.max_neurons = max_neurons

    def mutate_network(self, neural_network):
        number_layers = len(neural_network.hidden_layers)
        neurons = self.max_neurons if self.max_neurons >= number_layers else number_layers
        neurons = sample(range(1, neurons), number_layers - 1)
        hidden_layers = [[create_neuron() for i in range(neuron)] for neuron in neurons]
        hidden_layers.append([self._create_final_hidden_neuron()])
        return hidden_layers




