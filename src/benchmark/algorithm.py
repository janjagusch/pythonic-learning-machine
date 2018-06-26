from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.neat_python.algorithm import Neat
from algorithms.simple_genetic_algorithm.algorithm import SimpleGeneticAlgorithm
from timeit import default_timer

_time_seconds = lambda: default_timer()

def _get_parent(algorithm):
    return algorithm.__class__.__bases__[0]

def _benchmark_fit(algorithm, input_matrix, target_vector, metric, verbose):
    parent = _get_parent(algorithm)
    parent.fit(algorithm, input_matrix, target_vector, metric, verbose)
    return algorithm.log

def _benchmark_run(algorithm, verbose=False):
    parent = _get_parent(algorithm)
    time_log = list()
    solution_log = list()
    stopping_criterion = False
    while (not stopping_criterion):
        start_time = _time_seconds()
        stopping_criterion = parent._epoch(algorithm)
        end_time = _time_seconds()
        time_log.append(end_time - start_time)
        solution_log.append(algorithm.champion)
        if verbose:
            parent._print_generation(algorithm)
        algorithm.current_generation += 1
        algorithm.log = {
        'time_log': time_log,
        'solution_log': solution_log}

class BenchmarkSLM(SemanticLearningMachine):

    fit = _benchmark_fit
    _run = _benchmark_run

class BenchmarkNEAT(Neat):

    fit = _benchmark_fit
    _run = _benchmark_run

class BenchmarkSGA(SimpleGeneticAlgorithm):

    fit = _benchmark_fit
    _run = _benchmark_run
