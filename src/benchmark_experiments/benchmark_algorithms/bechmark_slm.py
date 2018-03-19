from algorithms.semantic_learning_machine.semantic_learning_machine import SemanticLearningMachine
from timeit import default_timer
from copy import deepcopy

class BenchmarkSLM(SemanticLearningMachine):

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.log = self._run()
        self.current_champion.neural_network = deepcopy(self.current_champion.neural_network)
        self.X = None
        self.y = None

    def _run(self):
        time_log = list()
        solution_log = list()
        time_seconds = lambda: default_timer()
        stopping_criterion = False
        while(not stopping_criterion):
            run_start = time_seconds()
            stopping_criterion = self._epoch()
            run_end = time_seconds()
            time_log.append(run_end - run_start)
            solution_log.append(self.current_champion)
        return {
            'time_log': time_log,
            'solution_log': solution_log
        }

    def get_meta(self):
        pass
        # Training error evolution
        # Testing error evolution
        # Number of layers
        # Number of neurons
        # Number of connections
        # Processing time


