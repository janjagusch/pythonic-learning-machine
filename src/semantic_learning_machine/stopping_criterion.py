from numpy import var

class StoppingCriterion(object):

    def stop_evolution(self, algorithm):
        pass

class MaxGenerationsCriterion(StoppingCriterion):
    """Stops evolutionary process, if current generation of algorithm is greater than max generation."""

    def __init__(self, max_generation):
        self.max_generation = max_generation

    def stop_evolution(self, algorithm):
        if algorithm.current_generation > self.max_generation:
            return True
        else:
            return False

class ErrorDeviationVariationCriterion(StoppingCriterion):
    """Stops evolutionary process, if the share of solutions with lower error deviation variation amongst the
    superior offspring is less than defined threshold."""

    def __init__(self, threshold):
        self.threshold = threshold

    def stop_evolution(self, algorithm):

        # Subsets offspring that are better than ancestor.
        champion = algorithm.champion
        superior_offspring = [offspring for offspring in algorithm.population
                              if offspring.mean_error < champion.mean_error]

        # Calculates variance of remaining solutions.
        superior_var = [var(offspring.absolute_error) for offspring in superior_offspring]
        champion_var = var(champion)

        # Subsets offspring that have a lower error deviation variation.
        lower_var = [var for var in superior_var if var < champion_var]

        # Calculates percentage.
        percentage_lower = len(lower_var) / len(superior_var)

        # If percentage lower than threshold, return True, else False.
        if percentage_lower < self.threshold:
            return True
        else:
            return False