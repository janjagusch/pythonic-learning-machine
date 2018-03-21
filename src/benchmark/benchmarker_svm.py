# from benchmark_experiments.benchmark_configuration import SVC_CONFIGURATIONS, SVR_CONFIGURATIONS
# from data.data_set import is_classification, get_input_variables, get_target_variable
# from sklearn.svm import SVC, SVR
# from utils.calculations import root_mean_squared_error
#
#
# def svm_benchmark(training, validation, testing):
#     """"""
#     # Validation error for all configurations.
#     validation_error_list = list()
#     # Best model.
#     best_model = None
#     # Best configuration (lowest validation error).
#     best_configuration = None
#     # Best (lowest) validation error.
#     best_validation_error = float('Inf')
#
#     SVM_CONFIGURATIONS = SVC_CONFIGURATIONS if is_classification(training) else SVR_CONFIGURATIONS
#     SVM = SVC if is_classification(training) else SVR
#     # Iterate through all configurations.
#     for svm_configuration in SVM_CONFIGURATIONS:
#         model = SVM(**svm_configuration, cache_size=1000)
#         model.fit(get_input_variables(training).as_matrix(), get_target_variable(training).as_matrix())
#         validation_error = _calculate_error(model, validation)
#         if validation_error < best_validation_error:
#             best_configuration = svm_configuration
#             best_validation_error = validation_error
#             best_model = model
#         validation_error_list.append((svm_configuration, validation_error))
#
#     training_error = _calculate_error(best_model, training)
#     testing_error = _calculate_error(best_model, testing)
#     return training_error, validation_error_list, testing_error
#
# def _get_predictions(model, data_set):
#     return model.predict(get_input_variables(data_set).as_matrix())
#
# def _calculate_error(model, data_set):
#     predictions = _get_predictions(model, data_set)
#     target = get_target_variable(data_set)
#     return root_mean_squared_error(target - predictions)
