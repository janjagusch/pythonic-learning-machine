def _standardize_data_set(data_set):
    """

    Args:
        data_set:

    Returns:

    """
    return (data_set - data_set.mean()) / data_set.std()