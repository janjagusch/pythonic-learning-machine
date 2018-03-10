from numpy import apply_along_axis, exp, tanh


def calculate_output_identity(input_sum_array):
    """
    Applies identity function to input.
    Args:
        input_sum_array: Float value, corresponding to weighted sum of connections.

    Returns:
        Float value, corresponding to output.
    """
    return input_sum_array


def calculate_output_sigmoid(input_sum_array):
    """
    Applies sigmoid function to input.
    Args:
        input_sum_array: Float value, corresponding to weighted sum of connections.

    Returns:
        Float value, corresponding to output.
    """
    return 1 / (1 + exp(-input_sum_array))


def calculate_output_tanh(input_sum):
    """
    Applies sigmoid function to input.
    Args:
        input_sum: Float value, corresponding to weighted sum of connections.

    Returns:
        Float value, corresponding to output.
    """
    return tanh(input_sum)


def calculate_output(input_sum, activation_function_id):
    """

    Args:
        input_sum:
        activation_function_id:

    Returns:

    """
    activation_function = ACTIVATION_FUNCTIONS.get(activation_function_id)
    return activation_function(input_sum)


ACTIVATION_FUNCTIONS = {'identity': calculate_output_identity,
                        'sigmoid': calculate_output_sigmoid,
                        'tanh': calculate_output_tanh}
