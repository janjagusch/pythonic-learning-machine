from numpy import sqrt


def root_mean_squared_error(abs_error_array):
    return sqrt((abs_error_array ** 2).mean())
