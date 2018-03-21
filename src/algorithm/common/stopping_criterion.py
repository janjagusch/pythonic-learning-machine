from numpy import var

class StoppingCriterion(object):

    def evaluate(self, algorithm):
        return self.evaluate_final(algorithm)

    def evaluate_final(self, algorithm):
        """Makes sure that all algorithm terminate."""
        return algorithm.current_generation >= 200

class MaxGenerationsCriterion(StoppingCriterion):
    """Stops evolutionary process, if current generation of algorithm is greater than max generation."""

    def __init__(self, max_generation):
        self.max_generation = max_generation

    def evaluate(self, algorithm):
        return algorithm.current_generation >= self.max_generation

class ErrorDeviationVariationCriterion(StoppingCriterion):
    """Stops evolutionary process, if the share of solutions with lower error deviation variation amongst the
    superior offspring is less than defined threshold."""

    def __init__(self, threshold):
        self.threshold = threshold

    def evaluate(self, algorithm):

        # If current generation is 0, return False (since there exists no champion).
        if algorithm.current_generation == 0:
            return False
        champion = algorithm.champion
        # Subsets offspring that are better than ancestor.
        superior_solutions = [solution for solution in algorithm.population if solution.better_than_ancestor]
        # If not superior offspring exist, determine parent stopping criterion.
        if not superior_solutions:
            return super().evaluate(algorithm)
        # Calculate error variance for champion.
        var_champion = var(champion.predictions - algorithm.target_vector)
        # Calculate error variance for offspring.
        var_superior_solutions = [var(superior_solution.predictions - algorithm.target_vector) for superior_solution in superior_solutions]
        # Subsets offspring that have a lower error deviation variation.
        lower_var = [var for var in var_superior_solutions if var < var_champion]
        # Calculates percentage.
        percentage_lower = len(lower_var) / len(var_superior_solutions)
        # If percentage lower than threshold, return True, else evaluate parent.
        if percentage_lower < self.threshold:
            return True
        else:
            return super().evaluate(algorithm)