from data.io_data_set import load_samples
from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorSGA
from benchmark.configuration import SLM_FLS_CONFIGURATIONS, SLM_OLS_CONFIGURATIONS, \
    NEAT_CONFIGURATIONS, SGA_CONFIGURATIONS
from algorithm.common.metric import RootMeanSquaredError
from tqdm import tqdm

_MODELS = {
    'slm_fls': {
        'name_long': 'Semantic Learning Machine (Fixed Learning Step)',
        'name_short': 'SLM (FLS)',
        'algorithm': EvaluatorSLM,
        'configurations': SLM_FLS_CONFIGURATIONS},
    'slm_ols': {
        'name_long': 'Semantic Learning Machine (Optimized Learning Step)',
        'name_short': 'SLM (OLS)',
        'algorithm': EvaluatorSLM,
        'configurations': SLM_OLS_CONFIGURATIONS},
    'neat': {
        'name_long': 'Neuroevolution of Augmenting Topologies',
        'name_short': 'NEAT',
        'algorithm': EvaluatorNEAT,
        'configurations': NEAT_CONFIGURATIONS},
    'sga': {
        'name_long': 'Simple Genetic Algorithm',
        'name_short': 'SGA',
        'algorithm': EvaluatorSGA,
        'configurations': SGA_CONFIGURATIONS},
}

class Benchmarker(object):

    def __init__(self, data_set_name, metric=RootMeanSquaredError, models=_MODELS):
        self.samples = [load_samples(data_set_name, index) for index in range(30)]
        self.metric = metric
        self.models = models

    def _evaluate_algorithm(self, algorithm, configurations, training_set, validation_set, testing_set, metric):
        evaluator = algorithm(configurations, training_set, validation_set, testing_set, metric)
        return evaluator.run()

    def run(self):
        benchmark_results = {k:[] for k in self.models.keys()}

        for training, validation, testing in tqdm(self.samples):
            # print_progress(iteration, total)
            for key, value in tqdm(self.models.items()):
                # print_progress(sub_iteration, sub_total)
                benchmark_results[key].append(self._evaluate_algorithm(
                    algorithm=value['algorithm'], configurations=value['configurations'], training_set=training,
                    validation_set=validation, testing_set=testing, metric=self.metric
                ))
