from os import remove
from os.path import join, dirname

def create_configuration(configuration_dict):
    configuration = \
"""[NEAT]
fitness_criterion = mean
fitness_threshold = 10
no_fitness_termination = True
pop_size              = {population_size}
reset_on_extinction   = True

[DefaultSpeciesSet]
compatibility_threshold = {compatibility_threshold}

[DefaultStagnation]
species_fitness_func = mean
max_stagnation       = 15
species_elitism      = 1

[DefaultReproduction]
elitism            = 1
survival_threshold = 0.2

[DefaultGenome]
activation_default      = random
activation_mutate_rate  = 0.01
activation_options      = sigmoid identity relu tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.1
bias_mutate_rate        = 0.01
bias_replace_rate       = 0.01

compatibility_disjoint_coefficient = {compatibility_disjoint_coefficient}
compatibility_weight_coefficient   = {compatibility_weight_coefficient}

conn_add_prob           = {conn_add_prob}
conn_delete_prob        = {conn_delete_prob}

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = partial_nodirect 0.25

node_add_prob           = {node_add_prob}
node_delete_prob        = {node_delete_prob}

num_hidden              = 1
num_inputs              = {num_inputs}
num_outputs             = 1

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

structural_mutation_surer = True

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1""".format(**configuration_dict)

    return configuration


def write_configuration(configuration):
    with open(get_configuration_path(), 'w') as configuration_file:
        print(configuration, file=configuration_file)


def remove_configuration():
    remove(get_configuration_path())


def get_configuration_path():
    return join(dirname(__file__), 'configuration.ini')