from src.data.io_data_set import remove_extension, list_files, read_standardized_data_set, data_set_to_pickle
from os.path import join


def _resample_data_set(data_set, train_frac=None, valid_frac=None, test_frac=None):
    """"""
    assert len([frac for frac in [train_frac, valid_frac, test_frac] if not frac]) <= 1
    assert sum([frac for frac in [train_frac, valid_frac, test_frac] if frac]) <= 1

    if not train_frac:
        train_frac = 1 - (valid_frac + test_frac)
    if not valid_frac:
        valid_frac = 1 - (train_frac + test_frac)
    if not test_frac:
        test_frac = 1 - (train_frac + valid_frac)

    valid_frac_cond = valid_frac / (train_frac + valid_frac)

    test_data = data_set.sample(frac=test_frac).sort_index()
    train_data = data_set.drop(test_data.index)
    validation_data = train_data.sample(frac=valid_frac_cond).sort_index()
    train_data = train_data.drop(validation_data.index)

    return train_data, validation_data, test_data

def _resample_data_set_iter(data_set, train_frac=None, valid_frac=None, test_frac=None, iterations=30):
    """"""
    return list(_resample_data_set(data_set, train_frac, valid_frac, test_frac) for i in range(iterations))

def _to_pickle_sample(data_samples, data_set_name, index):

    file_path = join('04_resampled', data_set_name)

    file_names = [sample_name + "_" + str(index) for sample_name in ['training', 'validation', 'testing']]

    [data_set_to_pickle(data_sample, file_path, file_name) for data_sample, file_name in zip(data_samples, file_names)]

def _to_pickle(data_samples_list, data_set_name):
    [_to_pickle_sample(data_samples, data_set_name, index) for data_samples, index
     in zip(data_samples_list, range(len(data_samples_list)))]

def process_all_data_sets():
    """"""
    data_set_names = [remove_extension(file) for file in list_files('03_standardized')]
    [_to_pickle(data_samples_list, data_set_name) for data_samples_list, data_set_name in
     zip(
         [_resample_data_set_iter(data_set, 0.5, 0.3, 0.2) for data_set in
          [read_standardized_data_set(data_set_name) for data_set_name in data_set_names]],
         data_set_names)]

process_all_data_sets()