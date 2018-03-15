from copy import deepcopy
from numpy import array, shape, resize
from simple_genetic_algorithm.neural_network.node import Neuron, Sensor
from simple_genetic_algorithm.neural_network.activation_function import _ACTIVATION_FUNCTIONS
from simple_genetic_algorithm.neural_network.connection import Connection
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

    def __deepcopy__(self, memodict={}):
        # Bias can be referenced.
        bias = self.bias
        # Add bias to memodict.
        memodict[id(bias)] = []
        # Sensors can be referenced.
        sensors = self.sensors
        # Deepcopy of hidden layers.
        hidden_layers = deepcopy(self.hidden_layers, memodict)
        # Deepcopy of output neuron.
        output_neuron = deepcopy(self.output_neuron, memodict)
        return NeuralNetwork(sensors, bias, hidden_layers, output_neuron)


    def calculate(self):
        """Calculates semantics of all hidden neurons and output neuron."""
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.calculate()
        self.output_neuron.calculate()

    def get_predictions(self):
        """Returns semantics of output neuron."""
        return self.output_neuron.semantics

    def load_sensors(self, data_set):
        """Loads input variables of data set into sensors. Adjusts length of bias."""
        for sensor, sensor_data in zip(self.sensors, get_input_variables(data_set)):
            sensor.semantics = data_set[sensor_data].as_matrix()
        # TODO: Make one neural network (also, pay attention to the resize function)
        # TODO: From node in connection is not a reference, its a copy.
        self.bias.semantics = resize(self.bias.semantics, shape(self.sensors[0].semantics))

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

    def wipe_semantics(self):
        for neuron in self.get_hidden_neurons(): neuron.semantics = array([])
        self.output_neuron.semantics = array([])

    def add_sensors(self, data_set):
        self.sensors = [Sensor(array([])) for sensor_data in get_input_variables(data_set)]
        self.load_sensors(data_set)
        _connect_nodes(self.sensors, self.hidden_layers[0])

def create_neuron(activation_function=None, bias=None):
    """Creates neuron with defined activation function and bias."""
    # If activation function not defined, choose activation function at random.
    if not activation_function:
        activation_function = choice(list(_ACTIVATION_FUNCTIONS.keys()))
    neuron = Neuron(array([]), list(), activation_function)
    # If is biased, connect to bias with random weight.
    if bias:
        Connection(bias, neuron, 0)
    return neuron

def _connect_nodes(from_nodes, to_nodes, weight=0):
    """Connects all from nodes with all to nodes with determined weight."""
    for to_node in to_nodes:
        for from_node in from_nodes:
            Connection(from_node, to_node, weight)

def create_network_from_topology(topology):
    """Creates neural network from topology."""
    # Create bias.
    bias = Sensor(array([1]))
    # Creates neurons from remaining items in string.
    hidden_layers = [[create_neuron('tanh', bias) for i in range(j)] for j in topology]
    # Create output neuron.
    output_neuron = create_neuron('identity', bias)
    # Connect nodes in neural network.
    if len(hidden_layers) > 1:
        for i in range(1, len(hidden_layers)): _connect_nodes(hidden_layers[i - 1], hidden_layers[i])
        _connect_nodes(hidden_layers[i], [output_neuron])
    else:
        _connect_nodes(hidden_layers[0], [output_neuron])
    # Create neural network.
    neural_network = NeuralNetwork(None, bias, hidden_layers, output_neuron)
    # Return neural network.
    return neural_network
