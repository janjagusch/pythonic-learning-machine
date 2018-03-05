class NeuralNetwork(object):
    """
    Class represents neural network.
    Attributes:
        layer_list: List of layers.
        output_neuron: Output neuron.
    """

    def __init__(self, layer_list, output_neuron):
        self.layer_list = layer_list
        self.output_neuron = output_neuron

    def get_neuron_layers(self):
        return self.layer_list[1:]

    def get_sensor_layer(self):
        return self.layer_list[0]

    def calculate_neurons(self):
        for neuron_layer in self.get_neuron_layers():
            for neuron in neuron_layer.nodes:
                neuron.calculate_neuron()