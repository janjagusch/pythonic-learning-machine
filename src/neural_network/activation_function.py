from math import exp, tanh


def calculate_output_identity(input_sum):
    """
    Applies identity function to input.
    Args:
        input_sum: Float value, corresponding to weighted sum of connections.

    Returns:
        Float value, corresponding to output.
    """
    output = input_sum
    return output


def calculate_output_sigmoid(input_sum):
    """
    Applies sigmoid function to input.
    Args:
        input_sum: Float value, corresponding to weighted sum of connections.

    Returns:
        Float value, corresponding to output.
    """
    output = 1 / (1 + exp(-input_sum))
    return output


def calculate_output_tanh(input_sum):
    """
    Applies sigmoid function to input.
    Args:
        input_sum: Float value, corresponding to weighted sum of connections.

    Returns:
        Float value, corresponding to output.
    """
    output = tanh(input_sum)
    return output


def calculate_output(input_sum, activation_function_id):
    """

    Args:
        input_sum:
        activation_function_id:

    Returns:

    """
    activation_function = activation_function_dict.get(activation_function_id)
    output = activation_function(input_sum)
    return output


activation_function_dict = {'identity': calculate_output_identity,
                            'sigmoid': calculate_output_sigmoid,
                            'tanh': calculate_output_tanh}
