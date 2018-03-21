from copy import copy, deepcopy
from numpy import array
from algorithm.common.neural_network.node import Neuron, Sensor
from algorithm.common.neural_network.activation_function import _ACTIVATION_FUNCTIONS
from algorithm.common.neural_network.connection import Connection
from random import choice, uniform


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
        bias = self.bias
        # Sensors can be referenced.
        sensors = self.sensors
        # Creates shallow copy of every hidden layer, while only referencing the contained neurons.
        hidden_layers = [copy(hidden_layer) for hidden_layer in self.hidden_layers] if self.hidden_layers else list()
        # Copies output neuron.
        output_neuron = copy(self.output_neuron) if self.output_neuron else None
        # Returns shallow copy of self.
        return NeuralNetwork(sensors, bias, hidden_layers, output_neuron)

    def __deepcopy__(self, memodict={}):
        bias = deepcopy(self.bias, memodict)
        sensors = deepcopy(self.sensors, memodict)
        hidden_layers = deepcopy(self.hidden_layers, memodict)
        output_neuron = deepcopy(self.output_neuron, memodict)
        neural_network = NeuralNetwork(sensors, bias, hidden_layers, output_neuron)
        memodict[id(self)] = neural_network
        return neural_network

    def get_weights(self):
        return array([connection.weight for connection in self.get_connections()])

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
        return {
            'layers': len(self.hidden_layers),
            'neurons': len(self.get_hidden_neurons()),
            'connections': len(self.get_connections())
        }

    def predict(self, X):
        self.load_sensors(X)
        self.calculate()
        return self.get_predictions()

    def add_sensors(self, X):
        """Adds sensors to neural network and loads input data."""
        self.sensors = [Sensor(array([])) for sensor_data in X.T]
        self.load_sensors(X)
        # Connects nodes to first level hidden layer.
        _connect_nodes(self.sensors, self.hidden_layers[0])

    def wipe_semantics(self):
        """Sets semantics of all neurons to empty numpy array."""
        for neuron in self.get_hidden_neurons(): neuron.semantics = array([])
        self.output_neuron.semantics = array([])

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