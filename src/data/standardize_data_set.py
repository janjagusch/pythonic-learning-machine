from src.data.io_data_set import read_cleaned_data_set, data_set_to_pickle
from os.path import join

def _standardize_data_set(data_set):
    """"""
    data_set_ext = remove_unary_features(data_set)

    return (data_set_ext - data_set_ext.mean()) / data_set_ext.std()

def remove_unary_features(data_set):
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


data_set_concrete = read_cleaned_data_set('r_bio')
data_set_concrete_std = _standardize_data_set(data_set_concrete)
_to_pickle(data_set_concrete_std, 'r_bio')