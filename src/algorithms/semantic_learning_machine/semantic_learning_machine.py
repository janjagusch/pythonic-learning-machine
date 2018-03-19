from algorithms.semantic_learning_machine.neural_network.node import Sensor
from numpy import array, matrix, dot, resize, shape
from numpy.linalg import pinv
from algorithms.semantic_learning_machine.neural_network.neural_network import NeuralNetwork, create_neuron
from algorithms.semantic_learning_machine.neural_network.connection import Connection
from random import uniform, sample, randint
from algorithms.semantic_learning_machine.solution import create_solution
from copy import copy, deepcopy


class SemanticLearningMachine(object):
    """"""
    def __init__(self, stopping_criterion, population_size, layers,
                 learning_step, max_connections, mutation_operator):
        self.stopping_criterion = stopping_criterion
        self.population_size = population_size
        self.layers = layers
        self.learning_step = learning_step
        self.max_connections = max_connections
        self.mutation_operator = mutation_operator
        self.population = list()
        self.current_champion = None
        self.next_champion = None
        self.current_generation = 0
        self.X = None
        self.y = None

    def _get_learning_step(self, partial_semantics):
        """"""
        if self.learning_step == 'optimized':
            return self._get_optimized_learning_step(partial_semantics)
        else:
            return self.learning_step

    def _get_optimized_learning_step(self, partial_semantics):
        """"""
        delta_target = copy(self.y)
        if self.current_champion:
            delta_target -= self.current_champion.neural_network.get_predictions()
        inverse = array(pinv(matrix(partial_semantics)))
        return dot(inverse.transpose(), delta_target)[0]

    def _get_connection_weight(self, weight):
        return weight if weight else uniform(-1, 1)

    def _connect_nodes(self, from_nodes, to_nodes, weight=None, random=False):
        if random:
            max_connections = self.max_connections if len(from_nodes) > self.max_connections else len(from_nodes)
            random_connections = randint(1, max_connections)
            from_nodes_sample = sample(from_nodes, random_connections)
        else:
            from_nodes_sample = from_nodes
        [[Connection(from_node, to_node, self._get_connection_weight(weight))
        for from_node in from_nodes_sample] for to_node in to_nodes]

    def _connect_nodes_mutation(self, hidden_layers):
        """Connects new mutation neurons to remainder of network."""

        # Sets reference to champion neural network.
        neural_network = self.current_champion.neural_network

        # Create hidden origin layer.
        from_layers = [copy(hidden_layer) for hidden_layer in hidden_layers]
        [hidden_layer_new.extend(hidden_layer_old) for hidden_layer_new, hidden_layer_old
         in zip(from_layers, neural_network.hidden_layers)]

        # Establish connections.
        try:
            self._connect_nodes(neural_network.sensors, hidden_layers[0])
        except TypeError:
            print(1)
        previous_neurons = from_layers[0]
        for from_layer, to_layer in zip(from_layers[1:], hidden_layers[1:]):
            self._connect_nodes(previous_neurons, to_layer)
            previous_neurons = from_layer

    def _initialize_sensors(self):
        return [Sensor(x) for x in self.X.T]

    def _initialize_bias(self, neural_network):
        return Sensor(resize(array([1]), shape(neural_network.sensors[0].semantics)))

    def _initialize_hidden_layers(self, neural_network):
        """"""
        # Create hidden layers with one neuron with random activation function each.
        hidden_layers = [[create_neuron(None, neural_network.bias)] for i in range(self.layers - 1)]
        # Add final hidden layer with one neuron with tanh activation function.
        hidden_layers.append([create_neuron('tanh', neural_network.bias)])
        return hidden_layers

    def _initialize_solution(self, neural_network):
        """"""

        # Create output neuron.
        neural_network.output_neuron = create_neuron('identity', None)

        # Create hidden layer.
        neural_network.hidden_layers = self._initialize_hidden_layers(neural_network)

        # Establish connections
        self._connect_nodes(neural_network.sensors, neural_network.hidden_layers[0], random=True)
        previous_neurons = neural_network.hidden_layers[0]
        for hidden_layer in neural_network.hidden_layers[1:]:
            self._connect_nodes(previous_neurons, hidden_layer, random=False)
            previous_neurons = hidden_layer

        # Calculate hidden neurons.
        [[neuron.calculate() for neuron in layer] for layer in neural_network.hidden_layers]

        # Get partial, hidden semantics.
        partial_semantics = neural_network.hidden_layers[-1][0].semantics

        # Connect last hidden neuron to output neuron.
        self._connect_nodes(previous_neurons, [neural_network.output_neuron],
                            self._get_learning_step(partial_semantics), random=False)

        # Calculate output semantics.
        neural_network.output_neuron.calculate()

        # Create solution.
        solution = create_solution(None, neural_network, self.y)

        # Return solution.
        return solution

    def _initialize_population(self):
        # Create sensors.
        sensors = self._initialize_sensors()

        # Create neural network.
        neural_network = NeuralNetwork(sensors, None, None, None)

        # Create bias.
        neural_network.bias = self._initialize_bias(neural_network)

        # Create initial population.
        for i in range(self.population_size):
            solution = self._initialize_solution(neural_network)
            if not self.next_champion:
                self.next_champion = solution
            elif solution.mean_error < self.next_champion.mean_error:
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)

    def _mutate_solution(self):
        # Creates shallow copy of champion.
        offspring = copy(self.current_champion)

        # Create hidden layers.
        hidden_layers = self.mutation_operator.mutate_network(self)

        # Extend hidden layers.
        [hidden_layer.extend(mutation_layer) for hidden_layer, mutation_layer
         in zip(offspring.neural_network.hidden_layers, hidden_layers)]


        # Connect hidden neurons to remainder of network.
        self._connect_nodes_mutation(hidden_layers)

        # Calculate hidden layer.
        for hidden_layer in hidden_layers:
            [neuron.calculate() for neuron in hidden_layer]

        # Connect final hidden neuron to output neuron.
        final_hidden_neuron = hidden_layers[-1][0]
        learning_step = self._get_learning_step(final_hidden_neuron.semantics)
        Connection(final_hidden_neuron, offspring.neural_network.output_neuron, learning_step)

        # Update semantics of output neuron
        offspring.neural_network.output_neuron.semantics += final_hidden_neuron.semantics * learning_step

        # Update error of solution
        offspring.error += final_hidden_neuron.semantics * learning_step
        offspring.mean_error = offspring.calculate_mean_error()
        offspring.better_than_ancestor = offspring.mean_error < self.current_champion.mean_error

        # After the output semantics are updated, we can remove the semantics from the final hidden neuron.
        final_hidden_neuron.semantics = None

        return offspring

    def _mutate_population(self):
        """"""
        for i in range(self.population_size):
            solution = self._mutate_solution()
            if not self.next_champion:
                if solution.mean_error < self.current_champion.mean_error:
                    self.next_champion = solution
                else:
                    solution.neural_network = None
            elif solution.mean_error < self.next_champion.mean_error:
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)

    def _wipe_population(self):
        self.population = list()

    def _override_current_champion(self):
        if self.next_champion:
            self.current_champion = self.next_champion
            self.next_champion = None

    def _epoch(self):
        if self.current_generation == 0:
            self._initialize_population()
        else:
            self._mutate_population()
        stopping_criterion = self.stopping_criterion.evaluate(self)
        self._override_current_champion()
        self._wipe_population()
        self.current_generation += 1
        return stopping_criterion

    def _run(self):
        stopping_criterion = False
        while(not stopping_criterion):
            stopping_criterion = self._epoch()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self._run()
        self.current_champion.neural_network = deepcopy(self.current_champion.neural_network)
        self.X = None
        self.y = None

    def predict(self, X):
        neural_network = self.current_champion.neural_network
        neural_network.load_sensors(X)
        neural_network.calculate()
        return neural_network.get_predictions()
