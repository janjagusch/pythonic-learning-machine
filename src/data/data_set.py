def get_input_variables(data_set):
    """Returns independent variables."""
    target = data_set.columns[len(data_set.columns) - 1]
    return data_set.drop(target, axis=1)


def get_target_variable(data_set):
    """Returns target variable."""
    return data_set[data_set.columns[len(data_set.columns) - 1]]
