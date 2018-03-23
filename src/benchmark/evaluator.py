from random import shuffle
from data.data_set import get_input_variables, get_target_variable
from algorithm.common.metric import is_better
from timeit import default_timer
from benchmark.algorithm import BenchmarkSLM, BenchmarkNEAT, BenchmarkSGA
from neat.nn import FeedForwardNetwork
from numpy import append, array
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from algorithm.common.ensemble import Ensemble

TIME_LIMIT_SECONDS = 30
TIME_BUFFER = 0.1

class Evaluator(object):

    def __init__(self, model, configurations, training_set, validation_set, testing_set, metric):
        self.model = model
        self.configurations = configurations
        self.training_set = training_set
        self.validation_set = validation_set
        self.testing_set = testing_set
        self.metric = metric

    def _get_learner_meta(self, learner):
        learner_meta = {
            'testing_value': self._calculate_value(learner, self.testing_set)
        }
        return learner_meta

    def _select_best_learner(self, time_limit=TIME_LIMIT_SECONDS, time_buffer=TIME_BUFFER, verbose=False):
        # Best learner found (lowest validation error).
        best_learner = None
        # Lowest validation error found.
        best_validation_value = float('-Inf') if self.metric.greater_is_better else float('Inf')
        # Validation error list.
        validation_value_list = list()
        # Current time in seconds.
        time_seconds = lambda: default_timer()
        # Random order of configurations.
        shuffle(self.configurations)
        # Number of configurations run.
        number_of_runs = 0
        # Start of run.
        run_start = time_seconds()
        # Time left.
        time_left = lambda: time_limit - (time_seconds() - run_start)
        # Iterate though all configurations.
        for configuration in self.configurations:
            # Create learner from configuration.
            learner = self.model(**configuration)
            # Train learner.
            if self.__class__.__bases__[0] == EvaluatorSklearn:
                learner.fit(get_input_variables(self.training_set).as_matrix(),
                            get_target_variable(self.training_set).as_matrix())
            else:
                learner.fit(get_input_variables(self.training_set).as_matrix(), get_target_variable(self.training_set).as_matrix(),
                        self.metric, verbose)
            # Calculate validation value.
            validation_value = self._calculate_value(learner, self.validation_set)
            # If validation error lower than best validation error, set learner as best learner and validation error as best validation error.
            if is_better(validation_value, best_validation_value, self.metric):
                best_learner = learner
                best_validation_value = validation_value
            # Add configuration and validation error to validation error list.
            validation_value_list.append((configuration, validation_value))
            # Increase number of runs.
            number_of_runs += 1
            # Calculate time left.
            run_end = time_left()
            # Calculate time expected for next run.
            run_expected = (time_limit - run_end) / number_of_runs
            # If no time left or time expected for next run is greater than time left, break.
            if run_end < 0 or run_end * (1+time_buffer) < run_expected:
                break
        # When all configurations tested, return best learner.
        return {
            'best_learner': best_learner,
            'validation_value_list': validation_value_list
        }

    def _calculate_value(self, learner, data_set):
        prediction = learner.predict(get_input_variables(data_set).as_matrix())
        target = get_target_variable(data_set).as_matrix()
        return self.metric.evaluate(prediction, target)

    def run(self, time_limit=TIME_LIMIT_SECONDS, time_buffer=TIME_BUFFER, verbose=False):
        log = self._select_best_learner(time_limit, time_buffer, verbose)
        learner_meta = self._get_learner_meta(log['best_learner'])
        learner_meta['validation_value_list'] = log['validation_value_list']
        return learner_meta

class EvaluatorSLM(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(BenchmarkSLM, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = learner.champion.value
        learner_meta['training_value_evolution'] = self._get_training_value_evolution(learner)
        learner_meta['testing_value_evolution'] = self._get_testing_value_evolution(learner)
        learner_meta['processing_time'] = self._get_processing_time(learner)
        learner_meta['topology'] = self._get_topology(learner)
        return learner_meta

    def _get_solutions(self, learner):
        return learner.log['solution_log']

    def _get_training_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.value for solution in solutions]

    def _get_testing_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._calculate_network_value(solution.neural_network, self.testing_set) for solution in solutions]

    def _calculate_network_value(self, network, data_set):
        predictions = network.predict(get_input_variables(data_set).as_matrix())
        target = get_target_variable(data_set).as_matrix()
        return self.metric.evaluate(predictions, target)

    def _get_processing_time(self, learner):
        return learner.log['time_log']

    def _get_topology(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.neural_network.get_topology() for solution in solutions]

class EvaluatorNEAT(Evaluator):
    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(BenchmarkNEAT, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._get_solution_value(learner.champion)
        learner_meta['training_value_evolution'] = self._get_training_value_evolution(learner)
        learner_meta['testing_value_evolution'] = self._get_testing_value_evolution(learner)
        learner_meta['processing_time'] = self._get_processing_time(learner)
        learner_meta['topology'] = self._get_topology(learner)
        return learner_meta

    def _get_solutions(self, learner):
        return learner.log['solution_log']

    def _get_solution_value(self, solution):
        return solution.fitness if self.metric.greater_is_better else 1 / solution.fitness

    def _get_training_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._get_solution_value(solution) for solution in solutions]

    def _get_testing_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._calculate_solution_value(solution, self.testing_set, learner) for solution in solutions]

    def _calculate_solution_value(self, solution, data_set, learner):
        X = get_input_variables(data_set).as_matrix()
        target = get_target_variable(data_set).as_matrix()
        neural_network = FeedForwardNetwork.create(solution, learner.configuration)
        prediction = self._predict_neural_network(neural_network, X)
        return self.metric.evaluate(prediction, target)

    def _predict_neural_network(self, neural_network, X):
        predictions = array([])
        for data in X:
            predictions = append(predictions, float(neural_network.activate(data)[0]))
        return predictions

    def _get_processing_time(self, learner):
        return learner.log['time_log']

    def _get_topology(self, learner):
        solutions = self._get_solutions(learner)
        return [self._get_genome_topology(solution) for solution in solutions]

    def _get_genome_topology(self, genome):
        return {
            'neurons': len(genome.nodes),
            'connections': len(genome.connections)
        }

class EvaluatorSGA(EvaluatorSLM):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        Evaluator.__init__(self, BenchmarkSGA, configurations, training_set, validation_set, testing_set, metric)

class EvaluatorSklearn(Evaluator):

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._calculate_value(learner, self.training_set)
        return learner_meta

class EvaluatorSVC(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(SVC, configurations, training_set, validation_set, testing_set, metric)

class EvaluatorSVR(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(SVR, configurations, training_set, validation_set, testing_set, metric)

class EvaluatorMLPC(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(MLPClassifier, configurations, training_set, validation_set, testing_set, metric)

class EvaluatorMLPR(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(MLPRegressor, configurations, training_set, validation_set, testing_set, metric)

class EvaluatorRFC(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(RandomForestClassifier, configurations, training_set, validation_set, testing_set, metric)

class EvaluatorRFR(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(RandomForestRegressor, configurations, training_set, validation_set, testing_set, metric)

class EvaluatorEnsemble(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(Ensemble, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._calculate_value(learner, self.training_set)
        return learner_meta
