from copy import copy
from numpy import array, shape
from semantic_learning_machine.neural_network.node import Neuron
from semantic_learning_machine.neural_network.activation_function import ACTIVATION_FUNCTIONS
from semantic_learning_machine.neural_network.connection import Connection
from random import choice, uniform
from data.data_set import get_input_variables

class NeuralNetwork(object):
    """
    Class represents neural network.
    Attributes:
        hidden_layers: List of layers.
        output_neuron: Output neuron.
    """

    def __init__(self, sensors, bias, hidden_layers, output_neuron):
        self.sensors = sensors
        self.bias = bias
        self.hidden_layers = hidden_layers
        self.output_neuron = output_neuron

    def __copy__(self):
        copy_bias = self.bias
        copy_sensors = copy(self.sensors)
        copy_hidden_layers = [copy(hidden_layer) for hidden_layer in self.hidden_layers]
        copy_output_neuron = copy(self.output_neuron)
        return NeuralNetwork(copy_sensors, copy_bias, copy_hidden_layers, copy_output_neuron)

    def calculate(self):
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.calculate()
        self.output_neuron.calculate()

    def get_predictions(self):
        return self.output_neuron.semantics

    def load_sensors(self, data_set):
        for sensor, sensor_data in zip(self.sensors, get_input_variables(data_set)):
            sensor.semantics = data_set[sensor_data].as_matrix()
        self.bias.semantics.resize(shape(self.sensors[0].semantics), refcheck = False)

    def get_hidden_neurons(self):
        neurons = list()
        [neurons.extend(hidden_neurons) for hidden_neurons in self.hidden_layers]
        return neurons

    def get_connections(self):
        neurons = list()
        neurons.extend(self.get_hidden_neurons())
        neurons.append(self.output_neuron)
        connections = list()
        [connections.extend(neuron.input_connections) for neuron in neurons]
        return connections

    def get_topology(self):
        return (len(self.hidden_layers), len(self.get_hidden_neurons()), len(self.get_connections()))

def create_neuron(activation_function=None, bias=None):
    """"""
    # If activation function not defined, choose activation function at random.
    if not activation_function:
        activation_function = choice(list(ACTIVATION_FUNCTIONS.keys()))
    neuron = Neuron(array([]), list(), activation_function)
    # If is biased, connect to bias with random weight.
    if bias:
        Connection(bias, neuron, uniform(-1, 1))
    return neuron