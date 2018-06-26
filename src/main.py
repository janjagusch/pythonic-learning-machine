from benchmark.benchmarker import Benchmarker, continue_benchmark
from threading import Thread
from sys import argv
from time import sleep
import argparse
from data.io import get_benchmark_folder, read_pickle
from benchmark.formatter import format_benchmark
import os

def start_b(data_set_name, file_name=None):
    benchmarker = Benchmarker(data_set_name)
    benchmarker.run()


def continue_b(data_set_name, file_name):
    continue_benchmark(data_set_name, file_name)


if __name__ == '__main__':


    benchmark_paths = []

    for folder in os.listdir(get_benchmark_folder()):
        path = os.path.join(get_benchmark_folder(), folder)
        for file in os.listdir(path):
            benchmark_paths.append(os.path.join(get_benchmark_folder(), folder, file))

    for benchmark_path in benchmark_paths:
        benchmark = read_pickle(benchmark_path)
        benchmark_formatted = format_benchmark(benchmark)





    # parser = argparse.ArgumentParser(description='Runs benchmark for data set.')
    # parser.add_argument('-d', metavar='data_set_name', type=str, dest='data_set_name',
    #                     help='a name of a data set')
    # parser.add_argument('-f', metavar='file_name', type=str, dest='file_name',
    #                     help='a file name of an existing benchmark')
    #
    # args = parser.parse_args()
    #
    # if args.file_name:
    #     thread = Thread(target=continue_b, kwargs=vars(args))
    # else:
    #     thread = Thread(target=start_b, kwargs=vars(args))
    #
    # try:
    #     thread.daemon = True
    #     thread.start()
    #     while True: sleep(100)
    # except (KeyboardInterrupt, SystemExit):
    #     print('\n! Received keyboard interrupt, quitting threads.\n')
