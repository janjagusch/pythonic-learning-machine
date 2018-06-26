from algorithms.common.neural_network.node import Sensor
from algorithms.common.neural_network.neural_network import NeuralNetwork, create_neuron
from algorithms.common.neural_network.connection import Connection
from algorithms.semantic_learning_machine.solution import Solution
from algorithms.common.algorithm import EvolutionaryAlgorithm
from numpy import array, matrix, dot, resize, shape
from numpy.linalg import pinv
from random import uniform, sample, randint
from copy import copy, deepcopy


class SemanticLearningMachine(EvolutionaryAlgorithm):
    """
    Class represents Semantic Learning Machine (SLM) algorithms:
    https://www.researchgate.net/publication/300543369_Semantic_Learning_Machine_
    A_Feedforward_Neural_Network_Construction_Algorithm_Inspired_by_Geometric_Semantic_Genetic_Programming

    Attributes:
        layer: Number of layers for base topology.
        learning_step: Weight for connection to output neuron.
        max_connections: Maximum connections for neuron.
        mutation_operator: Operator that augments neural network.
        next_champion: Solution that will replace champion.

    Notes:
        learning_step can be positive numerical value of 'optimized' for optimized learning step.
    """

    def __init__(self, population_size, stopping_criterion,
                 layers, learning_step, max_connections, mutation_operator):
        super().__init__(population_size, stopping_criterion)
        self.layers = layers
        self.learning_step = learning_step
        self.max_connections = max_connections
        self.mutation_operator = mutation_operator
        self.next_champion = None

    def _get_learning_step(self, partial_semantics):
        """Returns learning step."""

        # If learning step is 'optimized', calculate optimized learning step.
        if self.learning_step == 'optimized':
            return self._get_optimized_learning_step(partial_semantics)
        # Else, return numerical learning step.
        else:
            return self.learning_step

    def _get_optimized_learning_step(self, partial_semantics):
        """Calculates optimized learning step."""

        # Calculates distance to target vector.
        delta_target = copy(self.target_vector).astype(float)
        if self.champion:
            delta_target -= self.champion.neural_network.get_predictions()
        # Calculates pseudo-inverse of partial_semantics.
        inverse = array(pinv(matrix(partial_semantics)))
        # Returns dot product between inverse and delta.
        return dot(inverse.transpose(), delta_target)[0]

    def _get_connection_weight(self, weight):
        """Returns connection weight if defined, else random value between -1 and 1."""

        return weight if weight else uniform(-1, 1)

    def _connect_nodes(self, from_nodes, to_nodes, weight=None, random=False):
        """
        Connects list of from_nodes with list of to_nodes.

        Args:
            from_nodes: List of from_nodes.
            to_nodes: List of to_nodes.
            weight: Weight from connection.
            random: Flag if random number of connections.

        Notes:
            If weight is None, then weight will be chosen at random between -1 and 1.
        """

        for to_node in to_nodes:
            # If random, create random sample of connection partners
            if random:
                max_connections = self.max_connections if len(from_nodes) > self.max_connections else len(from_nodes)
                random_connections = randint(1, max_connections)
                from_nodes_sample = sample(from_nodes, random_connections)
            else:
                from_nodes_sample = from_nodes
            # Connect to_node to each node in from_node_sample.
            for from_node in from_nodes_sample:
                Connection(from_node, to_node, self._get_connection_weight(weight))

    def _connect_nodes_mutation(self, hidden_layers):
        """Connects new mutation neurons to remainder of network."""

        # Sets reference to champion neural network.
        neural_network = self.champion.neural_network
        # Create hidden origin layer.
        from_layers = [copy(hidden_layer) for hidden_layer in hidden_layers]
        for hidden_layer_new, hidden_layer_old in zip(from_layers, neural_network.hidden_layers):
            hidden_layer_new.extend(hidden_layer_old)
        # Establish connections.
        self._connect_nodes(neural_network.sensors, hidden_layers[0], random=True)
        previous_neurons = from_layers[0]
        for from_layer, to_layer in zip(from_layers[1:], hidden_layers[1:]):
            self._connect_nodes(previous_neurons, to_layer, random=True)
            previous_neurons = from_layer

    def _connect_learning_step(self, neural_network):
        """Connects last hidden neuron with defined learning step."""

        # Get last hidden neuron.
        last_neuron = neural_network.hidden_layers[-1][-1]
        # Get semantics of last neuron.
        last_semantics = last_neuron.semantics
        # Connect last neuron to output neuron.
        self._connect_nodes([last_neuron], [neural_network.output_neuron], self._get_learning_step(last_semantics))

    def _create_solution(self, neural_network):
        """Creates solution for population."""

        # Creates solution object.
        solution = Solution(neural_network, None, None)
        # Calculates error.
        solution.value = self.metric.evaluate(neural_network.get_predictions(), self.target_vector)
        # Checks, if solution is better than parent.
        solution.better_than_ancestor = self._is_better_solution(solution, self.champion)
        # After the output semantics are updated, we can remove the semantics from the final hidden neuron.
        neural_network.output_neuron.input_connections[-1].from_node.semantics = None
        # Returns solution.
        return solution

    def _initialize_sensors(self):
        """Initializes sensors based on input matrix."""

        return [Sensor(input_data) for input_data in self.input_matrix.T]

    def _initialize_bias(self, neural_network):
        """Initializes biases with same length as sensors."""

        return Sensor(resize(array([1]), shape(neural_network.sensors[0].semantics)))

    def _initialize_hidden_layers(self, neural_network):
        """Initializes hidden layers, based on defined number of layers."""

        # Create hidden layers with one neuron with random activation function each.
        hidden_layers = [[create_neuron(None, neural_network.bias)] for i in range(self.layers - 1)]
        # Add final hidden layer with one neuron with tanh activation function.
        hidden_layers.append([create_neuron('tanh', neural_network.bias)])
        # Returns hidden layers.
        return hidden_layers

    def _initialize_topology(self):
        """Initializes topology."""

        # Create sensors.
        sensors = self._initialize_sensors()
        # Create neural network.
        neural_network = NeuralNetwork(sensors, None, None, None)
        # Create bias.
        neural_network.bias = self._initialize_bias(neural_network)
        # Return neural network.
        return neural_network

    def _initialize_neural_network(self, topology):
        """Creates neural network from initial topology."""

        # Create shallow copy of topology.
        neural_network = copy(topology)
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
        for layer in neural_network.hidden_layers:
            for neuron in layer:
                neuron.calculate()
        # Connect last neuron to output neuron with learning step.
        self._connect_learning_step(neural_network)
        # Calculate output semantics.
        neural_network.output_neuron.calculate()
        # Return neural network.
        return neural_network

    def _initialize_solution(self, topology):
        """Creates solution for initial population."""

        # Initialize neural network.
        neural_network = self._initialize_neural_network(topology)
        # Create solution.
        solution = self._create_solution(neural_network)
        # Return solution.
        return solution

    def _initialize_population(self):
        """Initializes population in first generation."""

        # Initializes neural network topology.
        topology = self._initialize_topology()
        # Create initial population from topology.
        for i in range(self.population_size):
            solution = self._initialize_solution(topology)
            if not self.next_champion:
                self.next_champion = solution
            elif self._is_better_solution(solution, self.next_champion):
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)

    def _mutate_network(self):
        """Creates mutated offspring from champion neural network."""

        # Create shallow copy of champion neural network.
        neural_network = copy(self.champion.neural_network)
        # Create mutated hidden layers.
        mutation_layers = self.mutation_operator.mutate_network(self)
        # Connect hidden neurons to remainder of network.
        self._connect_nodes_mutation(mutation_layers)
        # Calculate mutated hidden layer.
        for mutation_layer in mutation_layers:
            for neuron in mutation_layer:
                neuron.calculate()
        # Extend hidden layers.
        for hidden_layer, mutation_layers in zip(neural_network.hidden_layers, mutation_layers):
            hidden_layer.extend(mutation_layers)
        # Connect final hidden neuron to output neuron.
        self._connect_learning_step(neural_network)
        # Get most recent connection.
        connection = neural_network.output_neuron.input_connections[-1]
        # Update semantics of output neuron.
        neural_network.output_neuron.semantics += connection.from_node.semantics * connection.weight
        # Return neural network.
        return neural_network

    def _mutate_solution(self):
        """Applies mutation operator to current champion solution."""

        # Created mutated offspring of champion neural network.
        neural_network = self._mutate_network()
        # Create solution.
        solution = self._create_solution(neural_network)
        # Return solution.
        return solution

    def _mutate_population(self):
        """"""
        for i in range(self.population_size):
            solution = self._mutate_solution()
            if not self.next_champion:
                if self._is_better_solution(solution, self.champion):
                    self.next_champion = solution
                else:
                    solution.neural_network = None
            elif self._is_better_solution(solution, self.next_champion):
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)

    def _wipe_population(self):
        self.population = list()

    def _override_current_champion(self):
        if self.next_champion:
            self.champion = self.next_champion
            self.next_champion = None

    def _epoch(self):
        if self.current_generation == 0:
            self._initialize_population()
        else:
            self._mutate_population()
        stopping_criterion = self.stopping_criterion.evaluate(self)
        self._override_current_champion()
        self._wipe_population()
        return stopping_criterion

    def fit(self, input_matrix, target_vector, metric, verbose=False):
        super().fit(input_matrix, target_vector, metric, verbose)
        self.champion.neural_network = deepcopy(self.champion.neural_network)

    def predict(self, input_matrix):
        neural_network = self.champion.neural_network
        neural_network.load_sensors(input_matrix)
        neural_network.calculate()
        return neural_network.get_predictions()
