from copy import copy, deepcopy
from numpy import array, shape
from algorithms.semantic_learning_machine.neural_network.node import Neuron
from algorithms.semantic_learning_machine.neural_network.activation_function import _ACTIVATION_FUNCTIONS
from algorithms.semantic_learning_machine.neural_network.connection import Connection
from random import choice, uniform
from data.data_set import get_input_variables

class NeuralNetwork(object):
    """
    Class represents neural network.
    Attributes:
        sensors: List of input sensors.
        bias: Bias neuron.
        hidden_layers: List of layers, containing hidden neurons.
        output_neuron: Output neuron.
    """

    def __init__(self, sensors, bias, hidden_layers, output_neuron):
        self.sensors = sensors
        self.bias = bias
        self.hidden_layers = hidden_layers
        self.output_neuron = output_neuron

    def __copy__(self):
        # Bias can be referenced.
        copy_bias = self.bias
        # Sensors can be referenced.
        copy_sensors = self.sensors
        # Creates shallow copy of every hidden layer, while only referencing the contained neurons.
        copy_hidden_layers = [copy(hidden_layer) for hidden_layer in self.hidden_layers]
        # Copies output neuron.
        copy_output_neuron = copy(self.output_neuron)
        return NeuralNetwork(copy_sensors, copy_bias, copy_hidden_layers, copy_output_neuron)

    def __deepcopy__(self, memodict={}):
        bias = deepcopy(self.bias, memodict)
        sensors = deepcopy(self.sensors, memodict)
        hidden_layers = deepcopy(self.hidden_layers, memodict)
        output_neuron = deepcopy(self.output_neuron, memodict)
        neural_network = NeuralNetwork(sensors, bias, hidden_layers, output_neuron)
        memodict[id(self)] = neural_network
        return neural_network

    def calculate(self):
        """Calculates semantics of all hidden neurons and output neuron."""
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.calculate()
        self.output_neuron.calculate()

    def get_predictions(self):
        """Returns semantics of output neuron."""
        return self.output_neuron.semantics

    def load_sensors(self, X):
        """Loads input variables of data set into sensors. Adjusts length of bias."""
        for sensor, sensor_data in zip(self.sensors, X.T):
            sensor.semantics = sensor_data
        self.bias.semantics = array([1 for i in range(sensor_data.shape[0])])
        # self.bias.semantics.resize(shape(self.sensors[0].semantics), refcheck = False)

    def get_hidden_neurons(self):
        """Returns list of hidden neurons."""
        neurons = list()
        [neurons.extend(hidden_neurons) for hidden_neurons in self.hidden_layers]
        return neurons

    def get_connections(self):
        """Returns list of connections."""
        neurons = list()
        neurons.extend(self.get_hidden_neurons())
        neurons.append(self.output_neuron)
        connections = list()
        [connections.extend(neuron.input_connections) for neuron in neurons]
        return connections

    def get_topology(self):
        """Returns number of hidden layers, number of hidden neurons and number of connections."""
        return (len(self.hidden_layers), len(self.get_hidden_neurons()), len(self.get_connections()))

def create_neuron(activation_function=None, bias=None):
    """Creates neuron with defined activation function and bias."""
    # If activation function not defined, choose activation function at random.
    if not activation_function:
        activation_function = choice(list(_ACTIVATION_FUNCTIONS.keys()))
    neuron = Neuron(array([]), list(), activation_function)
    # If is biased, connect to bias with random weight.
    if bias:
        Connection(bias, neuron, uniform(-1, 1))
    return neuron