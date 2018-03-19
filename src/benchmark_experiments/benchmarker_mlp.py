from benchmark_experiments.parameter_tuner import MLP_CONFIGURATIONS
from sklearn.neural_network import MLPClassifier, MLPRegressor
from data.data_set import is_classification, get_input_variables, get_target_variable
from utils.calculations import root_mean_squared_error


def mlp_benchmark(training, validation, testing):
    """"""
    # Validation error for all configurations.
    validation_error_list = list()
    # Best model.
    best_model = None
    # Best configuration (lowest validation error).
    best_configuration = None
    # Best (lowest) validation error.
    best_validation_error = float('Inf')

    MLP = MLPClassifier if is_classification(training) else MLPRegressor

    i = 0

    # Iterate through all configurations.
    for mlp_configuration in MLP_CONFIGURATIONS:
        print(i)
        model = MLP(**mlp_configuration)
        model.fit(get_input_variables(training).as_matrix(), get_target_variable(training).as_matrix())
        validation_error = _calculate_error(model, validation)
        if validation_error < best_validation_error:
            best_configuration = mlp_configuration
            best_validation_error = validation_error
            best_model = model
        validation_error_list.append((mlp_configuration, validation_error))
        i += 1

    training_error = _calculate_error(best_model, training)
    testing_error = _calculate_error(best_model, testing)
    return training_error, validation_error_list, testing_error

def _get_predictions(model, data_set):
    return model.predict(get_input_variables(data_set).as_matrix())

def _calculate_error(model, data_set):
    predictions = _get_predictions(model, data_set)
    target = get_target_variable(data_set)
    return root_mean_squared_error(target - predictions)
