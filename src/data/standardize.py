from src.data.io_data_set import read_cleaned_data_set, data_set_to_pickle, remove_extension, list_files
from os.path import join
from data.data_set import is_classification

def _standardize_data_set(data_set):
    """"""
    data_set_ext = _remove_unary_features(data_set)
    cols = data_set_ext.columns
    if is_classification(data_set_ext):
        data_set_ext[cols[-1]] = data_set_ext[cols[-1]].astype(float)
        cols = cols[:-1]
    for c in cols:
        data_set_ext[c] = (data_set_ext[c] - data_set_ext[c].mean()) / data_set_ext[c].std()
    return data_set_ext

def _categorize_target_variable(data_set):
    data_set_ext = data_set.copy()
    target = data_set_ext[data_set_ext.columns[len(data_set_ext.columns) - 1]]
    if all(map(lambda x: x in (0, 1), target)):
        data_set_ext[data_set_ext.columns[len(data_set_ext.columns) - 1]] = target.astype('category')
    return data_set_ext

def _remove_unary_features(data_set):
    """"""
    data_set_ext = data_set.copy()
    for c in data_set_ext:
        column = data_set_ext[c]
        if all(map(lambda x: x == column[0], column)):
            data_set_ext.drop([c], axis=1, inplace=True)
    return data_set_ext

def _to_pickle(data_set, data_set_name):

    file_path = join('03_standardized')

    data_set_to_pickle(data_set, file_path, data_set_name)

def process_all_data_sets():
    """"""
    data_set_names = [remove_extension(file) for file in list_files('02_cleaned')]
    data_sets = [read_cleaned_data_set(data_set_name) for data_set_name in data_set_names]
    standardized_data_sets = [_standardize_data_set(data_set) for data_set in data_sets]
    [_to_pickle(standardized_data_set, data_set_name) for standardized_data_set, data_set_name in
     zip(standardized_data_sets, data_set_names)]


process_all_data_sets()