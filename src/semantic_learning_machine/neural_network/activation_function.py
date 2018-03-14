from numpy import exp, tanh, maximum


def calculate_identity(input_sum_array):
    """Applies identity function to input."""
    return input_sum_array

def calculate_sigmoid(input_sum_array):
    """Applies sigmoid function to input."""
    return 1 / (1 + exp(-input_sum_array))

def calculate_tanh(input_sum_array):
    """Applies hyperbolic tangent function to input."""
    return tanh(input_sum_array)

def calculate_relu(input_sum_array):
    """Applies rectified linear unit function to input."""
    return maximum(input_sum_array, 0, input_sum_array)

def calculate_output(input_sum_array, activation_function_id):
    """Applies activation function to input, based on determined id."""
    activation_function = _ACTIVATION_FUNCTIONS.get(activation_function_id)
    return activation_function(input_sum_array)

_ACTIVATION_FUNCTIONS = {'identity': calculate_identity,
                        'sigmoid': calculate_sigmoid,
                        'tanh': calculate_tanh,
                        'relu': calculate_relu}
