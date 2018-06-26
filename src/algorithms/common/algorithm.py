from algorithms.common.metric import is_better

class EvolutionaryAlgorithm(object):
    """
    Abstract class for evolutionary algorithms.

    Attributes:
        population_size: Number of individuals in population.
        metric: Evaluation metric of class Metric.
        population: List of individuals in current generation.
        chamption: Best individual found.
        current_generation: Current generation of evolutionary process.
        input_matrix: Data for sensors of neural network (numpy nd_array).
        target_vector: Target data for training during evolutionary process (numpy nd_array).
    """

    def __init__(self, population_size, stopping_criterion):
        self.population_size = population_size
        self.stopping_criterion = stopping_criterion
        self.metric = None
        self.population = list()
        self.champion = None
        self.current_generation = 0
        self.input_matrix = None
        self.target_vector = None

    def _print_generation(self):
        print('{}\t{:.2f}'.format(self.current_generation, self._get_champion_value()))

    def _get_champion_value(self):
        return self.champion.value

    def _is_better(self, value_1, value_2):
        """Returns whether value_1 is better than value_2, based on defined metric."""
        return is_better(value_1, value_2, self.metric)

    def _is_better_solution(self, solution_1, solution_2):
        if not solution_2:
            return None
        else:
            return self._is_better(solution_1.value, solution_2.value)

    def _evaluate(self, prediction, target):
        """Assigns value to difference between prediction and target, based on defined metric."""
        return self.metric.evaluate(prediction, target)

    def _epoch(self):
        pass

    def _run(self, verbose=False):
        stopping_criterion = False
        while(not stopping_criterion):
            stopping_criterion = self._epoch()
            if verbose:
                self._print_generation()
            self.current_generation += 1

    def fit(self, input_matrix, target_vector, metric, verbose=False):
        """Trains model to approximate y with X."""
        self.input_matrix = input_matrix
        self.target_vector = target_vector
        self.metric = metric
        self._run(verbose)
        self.input_matrix = None
        self.target_vector = None

    def predict(self, input_matrix):
        """Returns y predictions given X."""
        pass
