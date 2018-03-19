from data.io_data_set import load_samples
from benchmark_experiments.benchmarker_neat import neat_benchmark
from benchmark_experiments.benchmarker_slm import slm_benchmark
from benchmark_experiments.benchmarker_sga import sga_benchmark
from benchmark_experiments.benchmarker_svm import svm_benchmark
from benchmark_experiments.benchmarker_mlp import mlp_benchmark
from utils.format import print_progress
from benchmark_experiments.parameter_tuner import SLM_FLS_CONFIGURATIONS, SLM_OLS_CONFIGURATIONS


class Benchmarker(object):

    def __init__(self, data_set_name):
        self.samples = [load_samples(data_set_name, index) for index in range(30)]

    def run(self):
        slm_fls_results = list()
        slm_ols_results = list()
        neat_results = list()
        sga_results = list()
        mlp_results = list()
        svm_results = list()

        total = len(self.samples)
        iteration = 0

        for training, validation, testing in self.samples:
            print_progress(iteration, total)
            slm_fls_results.append(slm_benchmark(training, validation, testing, SLM_FLS_CONFIGURATIONS))
            slm_ols_results.append(slm_benchmark(training, validation, testing, SLM_OLS_CONFIGURATIONS))
            neat_results.append(neat_benchmark(training, validation, testing))
            sga_results.append(sga_benchmark(training, validation, testing))
            mlp_results.append(mlp_benchmark(training, validation, testing))
            svm_results.append(svm_benchmark(training, validation, testing))
            iteration += 1

        print(1)


benchmarker = Benchmarker('r_concrete')

benchmarker.run()