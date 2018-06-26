
class Solution(object):
    """"""
    def __init__(self, neural_network, value, better_than_mutation_ancestor, better_than_crossover_ancestors):
        self.neural_network = neural_network
        self.value = value
        self.better_than_mutation_ancestor = better_than_mutation_ancestor
        self.better_than_crossover_ancestors = better_than_crossover_ancestors

# def create_solution(mutation_ancestor, crossover_ancestors, neural_network, target, better_than_crossover_ancestors):
#
#
#     solution = Solution(neural_network, None, None, None)
#
#     solution.error = solution._calculate_error(target)
#     solution.mean_error = solution.calculate_mean_error()
#     if mutation_ancestor:
#         solution.better_than_mutation_ancestor = solution._better_than(mutation_ancestor)
#     if crossover_ancestors:
#         solution.better_than_crossover_ancestors = all(solution._better_than(ancestor) for ancestor in crossover_ancestors)
#     if better_than_crossover_ancestors is not None:
#         solution.better_than_crossover_ancestors = better_than_crossover_ancestors
#     return solution
#
