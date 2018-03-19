import unittest
from algorithms.semantic_learning_machine.neural_network import Sensor, Neuron
from algorithms.semantic_learning_machine.neural_network.connection import Connection
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
        self.neuron_1.calculate()
        assert_array_equal(self.neuron_1.semantics, array([8, 11, 14, 17, 20]))

    def test_calculate_neurons(self):
        self.neural_network.calculate()

        assert_array_equal(self.neural_network.output_neuron.semantics, array([8, 6, 4, 2, 0]))


if __name__ == '__main__':
    unittest.main()
