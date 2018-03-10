from numpy import sqrt


def root_mean_squared_error(predictions, targets):
    return sqrt(((predictions - targets) ** 2).mean())


def root_mean_squared_error(abs_error_array):
    return sqrt((abs_error_array ** 2).mean())
