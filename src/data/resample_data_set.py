from src.data.io_data_set import remove_extension, list_files, read_cleaned_data_set, data_set_to_pickle
from os.path import join


def _resample_data_set(data_set, train_frac=None, valid_frac=None, test_frac=None):
    """

    Args:
        data_set:
        train_frac:
        valid_frac:
        test_frac:

    Returns:

    """

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


def resample_data_set(data_set, train_frac=None, valid_frac=None, test_frac=None, iterations=30):
    """"""
    return list(_resample_data_set(data_set, train_frac, valid_frac, test_frac) for i in range(iterations))


def _serialize_data_samples(data_samples, data_set_name, index):

    file_path = join('04_resampled', data_set_name)

    file_names = [sample_name + "_" + str(index) for sample_name in ['training', 'validation', 'testing']]

    [data_set_to_pickle(data_sample, file_path, file_name) for data_sample, file_name in zip(data_samples, file_names)]


def serialize_samples_list(data_samples_list, data_set_name):
    [_serialize_data_samples(data_samples, data_set_name, index) for data_samples, index
     in zip(data_samples_list, range(len(data_samples_list)))]


def process_all_data_sets():
    """"""
    data_set_names = [remove_extension(file) for file in list_files('02_cleaned')]
    [serialize_samples_list(data_samples_list, data_set_name) for data_samples_list, data_set_name in
     zip(
         [resample_data_set(data_set, 0.5, 0.3, 0.2) for data_set in
          [read_cleaned_data_set(data_set_name) for data_set_name in data_set_names]],
         data_set_names)]
