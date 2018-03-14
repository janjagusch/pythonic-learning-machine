from data.io_data_set import load_samples
from src.benchmarker.benchmarker_neat import neat_benchmark
from benchmarker.benchmarker_slm import slm_benchmark


class Benchmarker(object):

    def __init__(self, data_set_name):
        self.samples = [load_samples(data_set_name, index) for index in range(30)]

    def run(self):
        for training, validation, testing in self.samples:
            slm_results = slm_benchmark(training, validation, testing)
            neat_results = neat_benchmark(training, validation, testing)

            # Do something with SLM.
            # Do something with NEAT.
            # Do something with SGA.








benchmarker = Benchmarker('r_concrete')

benchmarker.run()