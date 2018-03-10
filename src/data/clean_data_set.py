from src.data.io_data_set import read_raw_data_set

cancer_raw = read_raw_data_set('c_cancer')


def _clean_cancer_data_set(data_set):
    """Cleans the cancer_raw data_sets set."""
    clean_data_set = data_set.copy()


    # Remove ID variable
    clean_data_set.drop(clean_data_set.columns[0], axis=1, inplace=True)

    # Convert target variable to numerical
    clean_data_set[clean_data_set.columns[2]].astype('category')


    # Append target variable to end of data set


-_clean_cancer_data_set(cancer_raw)



