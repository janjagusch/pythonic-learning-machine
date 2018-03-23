def get_input_variables(data_set):
    """Returns independent variables."""
    target = data_set.columns[len(data_set.columns) - 1]
    return data_set.drop(target, axis=1)

def get_target_variable(data_set):
    """Returns target variable."""
    return data_set[data_set.columns[len(data_set.columns) - 1]]

def is_classification(data_set):
    target = data_set[data_set.columns[len(data_set.columns) - 1]]
    return all(map(lambda x: x in (0, 1), target))

