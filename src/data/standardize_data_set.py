from src.data.io_data_set import remove_extension, list_files, read_cleaned_data_set, data_set_to_pickle

def _standardize_data_set(data_set):
    """"""
    return (data_set - data_set.mean()) / data_set.std()