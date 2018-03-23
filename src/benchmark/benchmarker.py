from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorSGA, \
    EvaluatorSVC, EvaluatorSVR, EvaluatorMLPC, EvaluatorMLPR, EvaluatorRFC, EvaluatorRFR, EvaluatorEnsemble
from benchmark.configuration import SLM_FLS_CONFIGURATIONS, SLM_OLS_CONFIGURATIONS, \
    NEAT_CONFIGURATIONS, SGA_CONFIGURATIONS, SVC_CONFIGURATIONS, SVR_CONFIGURATIONS, MLP_CONFIGURATIONS, \
    RF_CONFIGURATIONS, ENSEMBLE_CONFIGURATIONS
from algorithm.common.metric import RootMeanSquaredError
from data.extract import is_classification
from data.io import load_samples, benchmark_to_pickle, benchmark_from_pickle
from tqdm import tqdm
import datetime


# Returns the current date and time.
_now = datetime.datetime.now()

# Default models to be compared.
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
    'svc': {
        'name_long': 'Support Vector Machine',
        'name_short': 'SVM',
        'algorithm': EvaluatorSVC,
        'configurations': SVC_CONFIGURATIONS},
    'svr': {
        'name_long': 'Support Vector Machine',
        'name_short': 'SVM',
        'algorithm': EvaluatorSVR,
        'configurations': SVR_CONFIGURATIONS},
    'mlpc': {
        'name_long': 'Multilayer Perceptron',
        'name_short': 'MLP',
        'algorithm': EvaluatorMLPC,
        'configurations': MLP_CONFIGURATIONS},
    'mlpr': {
        'name_long': 'Multilayer Perceptron',
        'name_short': 'MLP',
        'algorithm': EvaluatorMLPR,
        'configurations': MLP_CONFIGURATIONS},
    'slm_ensemble': {
        'name_long': 'Semantic Learning Machine Ensemble',
        'name_short': 'SLM (Ensemble)',
        'algorithm': EvaluatorEnsemble,
        'configurations': ENSEMBLE_CONFIGURATIONS},
    'rfc': {
        'name_long': 'Random Forest',
        'name_short': 'RF',
        'algorithm': EvaluatorRFC,
        'configurations': RF_CONFIGURATIONS},
    'rfr': {
        'name_long': 'Random Forest',
        'name_short': 'RF',
        'algorithm': EvaluatorRFR,
        'configurations': RF_CONFIGURATIONS}
}


class Benchmarker(object):
    """
    Class represents benchmark environment to compare different algorithms in various parameter configurations
    on a given data set and a defined performed metric.

    Attributes:
        data_set_name: Name of data set to study.
        metric: Performance measure to compare with.
        models: Dictionary of models and their corresponding parameter configurations.
    """

    def __init__(self, data_set_name, metric=RootMeanSquaredError, models=_MODELS):
        """Initializes benchmark environment."""

        self.data_set_name = data_set_name
        # Creates file name as combination of data set name and and date.
        self.file_name = self.data_set_name + "__" + _now.strftime("%Y_%m_%d__%H_%M_%S")
        # Loads samples into object.
        self.samples = [load_samples(data_set_name, index) for index in range(30)]
        self.metric = metric
        self.models = models
        # If data set is classification problem, remove regression models. Else, vice versa.
        if is_classification(self.samples[0][0]):
            if 'svr' in self.models.keys():
                del self.models['svr']
            if 'mlpr' in self.models['mlpr']:
                del self.models['mlpr']
            if 'rfr' in self.models['rfr']:
                del self.models['rfr']
        else:
            if 'svc' in self.models.keys():
                del self.models['svc']
            if 'mlpc' in self.models.keys():
                del self.models['mlpc']
            if 'rfc' in self.models.keys():
                del self.models['rfc']
        # Create results dictionary with models under study.
        self.results = {k: [None for i in self.samples] for k in self.models.keys()}
        # Serialize benchmark environment.
        benchmark_to_pickle(self)

    def _evaluate_algorithm(self, algorithm, configurations, training_set, validation_set, testing_set, metric):
        """Creates evaluator, based on algorithm and configurations."""

        evaluator = algorithm(configurations, training_set, validation_set, testing_set, metric)
        return evaluator.run()

    def run(self):
        """Runs benchmark study, where it evaluates every algorithm on every sample set."""

        i = 0
        for training, validation, testing in tqdm(self.samples):
            for key, value in tqdm(self.models.items()):
                # If evaluation for key, iteration pair already exists, skip this pair.
                if not self.results[key][i]:
                    self.results[key][i] = self._evaluate_algorithm(
                        algorithm=value['algorithm'], configurations=value['configurations'], training_set=training,
                        validation_set=validation, testing_set=testing, metric=self.metric)
                    # Serialize benchmark.
                    benchmark_to_pickle(self)
            i += 1


def continue_benchmark(data_set_name, file_name):
    """Loads a benchmark from .pkl file and continues run."""

    benchmark = benchmark_from_pickle(data_set_name, file_name)
    benchmark.run()
