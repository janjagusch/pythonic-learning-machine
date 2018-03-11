from copy import copy

class NeuralNetwork(object):
    """
    Class represents neural network.
    Attributes:
        hidden_layers: List of layers.
        output_neuron: Output neuron.
    """

    def __init__(self, sensors, hidden_layers, output_neuron):
        self.sensors = sensors
        self.hidden_layers = hidden_layers
        self.output_neuron = output_neuron

    def __copy__(self):
        copy_sensors = copy(self.sensors)
        copy_hidden_layers = [copy(hidden_layer) for hidden_layer in self.hidden_layers]
        copy_output_neuron = copy(self.output_neuron)
        return NeuralNetwork(copy_sensors, copy_hidden_layers, copy_output_neuron)

    def calculate(self):
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.calculate()
        self.output_neuron.calculate()

    def get_predictions(self):
        return self.output_neuron.semantics
