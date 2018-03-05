import unittest
from neural_network.node import Sensor, Neuron
from neural_network.connection import Connection
from neural_network.layer import Layer
from neural_network.neural_network import NeuralNetwork
from numpy import array
from numpy.testing import assert_array_equal


class TestNeurons(unittest.TestCase):
    def setUp(self):
        self.sensor_1 = Sensor(array([1, 2, 3, 4, 5]))
        self.sensor_2 = Sensor(array([6, 7, 8, 9, 10]))
        self.neuron_1 = Neuron(array([]), list(), 'identity')
        self.connection_1 = Connection(self.sensor_1, self.neuron_1, 2)
        self.connection_2 = Connection(self.sensor_2, self.neuron_1, 1)

    def test_calculate_output(self):
        self.neuron_1.calculate_neuron()
        assert_array_equal(self.neuron_1.semantics, array([8, 11, 14, 17, 20]))


class TestNeuralNetworks(unittest.TestCase):
    def setUp(self):

        self.sensor_1 = Sensor(array([1, 2, 3, 4, 5]))
        self.sensor_2 = Sensor(array([6, 7, 8, 9, 10]))

        self.neuron_1 = Neuron(array([]), list(), 'identity')
        self.neuron_2 = Neuron(array([]), list(), 'identity')
        self.neuron_3 = Neuron(array([]), list(), 'identity')

        self.output_neuron = Neuron(array([]), list(), 'identity')

        self.connection_1 = Connection(self.sensor_1, self.neuron_1, -1)
        self.connection_2 = Connection(self.sensor_2, self.neuron_2, 1)
        self.connection_3 = Connection(self.neuron_1, self.neuron_3, 2)
        self.connection_4 = Connection(self.neuron_2, self.neuron_3, 1)
        self.connection_5 = Connection(self.neuron_3, self.output_neuron, 2)

        self.input_layer = Layer([self.sensor_1, self.sensor_2])
        self.layer_1 = Layer([self.neuron_1, self.neuron_2])
        self.layer_2 = Layer([self.neuron_3])
        self.output_layer = Layer([self.output_neuron])

        self.neural_network = NeuralNetwork([self.input_layer, self.layer_1, self.layer_2, self.output_layer],
                                            self.output_neuron)

    def test_calculate_neurons(self):
        self.neural_network.calculate_neurons()

        assert_array_equal(self.neural_network.output_neuron.semantics, array([8, 6, 4, 2, 0]))


if __name__ == '__main__':
    unittest.main()
