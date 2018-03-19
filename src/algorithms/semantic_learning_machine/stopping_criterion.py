from numpy import var

class StoppingCriterion(object):

    def evaluate(self, algorithm):
        self.evaluate_final(algorithm)

    def evaluate_final(self, algorithm):
        """Makes sure that all algorithms terminate."""
        return algorithm.current_generation >= 200

class MaxGenerationsCriterion(StoppingCriterion):
    """Stops evolutionary process, if current generation of algorithm is greater than max generation."""

    def __init__(self, max_generation):
        self.max_generation = max_generation

    def evaluate(self, algorithm):
        if algorithm.current_generation >= self.max_generation:
            return True
        else:
            super().evaluate(algorithm)

class ErrorDeviationVariationCriterion(StoppingCriterion):
    """Stops evolutionary process, if the share of solutions with lower error deviation variation amongst the
    superior offspring is less than defined threshold."""

    def __init__(self, threshold):
        self.threshold = threshold

    def evaluate(self, algorithm):
        if algorithm.current_generation == 0:
            return False

        # Subsets offspring that are better than ancestor.
        champion = algorithm.current_champion
        superior_offspring = [offspring for offspring in algorithm.population if offspring.better_than_ancestor]
        if not superior_offspring:
            return super().evaluate(algorithm)

        # Calculates variance of remaining solutions.
        superior_var = [var(offspring.error) for offspring in superior_offspring]
        champion_var = var(champion.error)

        # Subsets offspring that have a lower error deviation variation.
        lower_var = [var for var in superior_var if var < champion_var]

        # Calculates percentage.
        percentage_lower = len(lower_var) / len(superior_var)

        # If percentage lower than threshold, return True, else False.
        if percentage_lower < self.threshold:
            return True
        else:
            return super().evaluate(algorithm)