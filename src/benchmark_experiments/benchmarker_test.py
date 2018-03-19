from random import shuffle
from data.data_set import get_input_variables, get_target_variable
from utils.calculations import root_mean_squared_error
from benchmark_experiments.parameter_tuner import SLM_OLS_CONFIGURATIONS
from src.data.io_data_set import load_samples
from timeit import default_timer

TIME_LIMIT_SECONDS = 600

class Evaluator(object):

    def __init__(self, model, configurations, training_set, validation_set, testing_set):
        self.model = model
        self.configurations = configurations
        self.training_set = training_set
        self.validation_set = validation_set
        self.testing_set = testing_set

    def _compare_learners(self, learner_1, learner_2):
        pass

    def _get_learner_meta(self, learner):
        pass

    def _select_best_learner(self, time_limit = TIME_LIMIT_SECONDS):
        best_learner = None
        best_learner_meta = None
        best_validation_error = float('Inf')

        time_seconds = lambda: default_timer()

        shuffle(self.configurations)

        number_of_runs = 0

        run_start = time_seconds()

        time_left = lambda: time_limit - (time_seconds() - run_start)

        for configuration in self.configurations:

            learner = self.model(**configuration)
            learner.fit(get_input_variables(self.training_set).as_matrix(), get_target_variable(self.training_set).as_matrix())
            validation_error = self._calculate_error(learner, self.validation_set)
            if validation_error < best_validation_error:
                best_learner = learner
                best_learner_meta = self._get_learner_meta(learner)
                best_validation_error = validation_error

            number_of_runs += 1

            run_end = time_left()

            run_expected = (time_limit - run_end) / number_of_runs

            if run_end < run_expected:
                return best_learner, best_learner_meta

        return best_learner, best_learner_meta

    def _calculate_error(self, learner, data_set):
        predictions = learner.predict(get_input_variables(data_set).as_matrix())
        target = get_target_variable(data_set).as_matrix()
        return root_mean_squared_error(target - predictions)

    def _evaluate_learner(self, learner):
        pass

    def run(self):
        best_learner = self._select_best_learner()
        return self._evaluate_learner(best_learner)

from benchmark_experiments.benchmark_algorithms.bechmark_slm import BenchmarkSLM

class EvaluatorSLM(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set):
        super().__init__(BenchmarkSLM, configurations, training_set, validation_set, testing_set)


training, validation, testing = load_samples('r_concrete', 0)

evaluator_slm = EvaluatorSLM(SLM_OLS_CONFIGURATIONS, training, validation, testing)
evaluator_slm.run()

