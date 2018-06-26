from numpy import mean, median, std, sqrt
from scipy.stats import sem
import pandas as pd
import numpy as np
from data.io import _get_path_to_data_dir
import os

def _metric_in_dict(metric, d):
    return metric in d[0].keys()


def _get_dictionaries_by_metric(results, metric):
    return {k: results[k] for k in results.keys() if _metric_in_dict(metric, results[k])}


def _get_values_from_dictionary(dictionary, metric):
    return [d[metric] for d in dictionary]


def _summarize_metric(metric, summarizer=mean):
    return [summarizer([m[i] for m in metric]) for i in range(len(metric[0]))]


def _format_static_table(results, metric):
    dictionaries = _get_dictionaries_by_metric(results, metric)
    values = {k: _get_values_from_dictionary(dictionaries[k], metric) for k in dictionaries.keys()}
    return pd.DataFrame.from_dict(values)


def _format_processing_time_table(results):
    dictionaries = _get_dictionaries_by_metric(results, 'processing_time')
    values = {k: _get_values_from_dictionary(dictionaries[k], 'processing_time') for k in dictionaries.keys()}
    for key, value in values.items():
        values[key] = [sum(item) for item in value]
    return pd.DataFrame.from_dict(values)


def _format_topology_table(results, component):
    dictionaries = _get_dictionaries_by_metric(results, 'topology')
    values = {k: _get_values_from_dictionary(dictionaries[k], 'topology') for k in dictionaries.keys()}
    values = {key: [item[-1] for item in value] for key, value in values.items()}
    values = {key: [item[component] for item in value] for key, value in values.items()}
    return pd.DataFrame.from_dict(values)


def _format_evo_table(results, metric):
    dictionaries = _get_dictionaries_by_metric(results, metric)
    values = {k: _get_values_from_dictionary(dictionaries[k], metric) for k in dictionaries.keys()}
    values = {key: [[item[i] for item in value if i < len(item)]
                    for i in range(max([len(item) for item in value]))] for key, value in values.items()}

    max_len = max(len(value) for key, value in values.items())

    mean_dict = {key: [mean(item) for item in value] for key, value in values.items()}

    se_dict = {key: [std(item) / sqrt(len(item)) for item in value] for key, value in values.items()}

    for key, value in mean_dict.items():
        delta_len = max_len - len(value)
        mean_dict[key].extend([np.nan for i in range(delta_len)])

    for key, value in se_dict.items():
        delta_len = max_len - len(value)
        se_dict[key].extend([np.nan for i in range(delta_len)])

    return pd.DataFrame.from_dict(mean_dict), pd.DataFrame.from_dict(se_dict)


def format_results(results):
    formatted_results = {}
    formatted_results['training_value'] = _format_static_table(results, 'training_value')
    formatted_results['testing_value'] = _format_static_table(results, 'testing_value')
    formatted_results['processing_time'] = _format_processing_time_table(results)
    formatted_results['number_neurons'] = _format_topology_table(results, 'neurons')
    formatted_results['number_connections'] = _format_topology_table(results, 'connections')
    formatted_results['training_value_evolution'] = _format_evo_table(results, 'training_value_evolution')
    formatted_results['testing_value_evolution'] = _format_evo_table(results, 'testing_value_evolution')
    formatted_results['processing_time_evolution'] = _format_evo_table(results, 'processing_time')

    return formatted_results


def relabel_model_names(model_names, model_names_dict, short=True):
    key = 'name_short' if short else 'name_long'
    return [model_names_dict[model_name][key] for model_name in model_names]


def format_benchmark(benchmark):

    output_path = os.path.join(_get_path_to_data_dir(), '06_formatted', benchmark.data_set_name)

    # If 'file_path_ext' does not exist, create 'file_path_ext'.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if _is_classification(benchmark):
        del benchmark.results['mlpr']
        del benchmark.results['rfr']

    formatted_benchmark = format_results(benchmark.results)

    model_names_dict = get_model_names_dict(benchmark)
    for key, value in formatted_benchmark.items():
        if 'evolution' in key:
            i = 0
            for tbl in value:
                if i == 0:
                    ext = 'mean'
                else:
                    ext = 'se'
                tbl.columns = relabel_model_names(tbl.columns, model_names_dict)
                path = os.path.join(output_path, key + '_' + ext + '.csv')
                tbl.to_csv(path)
                i += 1
        else:
            formatted_benchmark[key].columns = relabel_model_names(value.columns, model_names_dict)
            path = os.path.join(output_path, key + '.csv')
            formatted_benchmark[key].to_csv(path)


def _is_classification(benchmark):
    return benchmark.data_set_name[0] == 'c'


def get_model_names_dict(benchmark):
    return {key: {'name_short': value['name_short'],
                  'name_long': value['name_long']} for key, value in benchmark.models.items()}
