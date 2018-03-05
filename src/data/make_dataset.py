from os.path import join, dirname
from os import pardir
from numpy import genfromtxt, mean, std
from subprocess import check_output


def _get_path_to_data_set(data_set_name):
    """
    Returns file path to data set.
    Args:
        data_set_name: Name of data set.

    Returns:
        File path to data set.
    """
    return join(dirname(__file__), pardir, pardir, "data", "raw", data_set_name, "data_set.csv")


def _standardize_data_set(data_set):
    """

    Args:
        data_set:

    Returns:

    """
    return (data_set - data_set.mean()) / data_set.std()


def make_credit_data_set():
    """
    Creates 'c_cancer' data set.
    Returns:
        Numpy array containing 'c_cancer' data.
    """
    file_path = _get_path_to_data_set("c_credit")
    data_set = genfromtxt(file_path, delimiter=',')
    return data_set

# credit_data_set = make_credit_data_set()
# print(credit_data_set)

# Define command and arguments
command = 'Rscript'
path2script = 'C:/Users/Jan/Documents/GitHub/pythonic-learning-machine/data/raw/c_cancer/c_cancer_preproc.R'

# Build subprocess command
cmd = [command, path2script]

# check_output will run the command and store to result
check_output(cmd, universal_newlines=True)
