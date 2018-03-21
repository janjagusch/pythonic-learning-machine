from random import choices, sample
from algorithm.simple_genetic_algorithm.solution import Solution
from operator import attrgetter

class SelectionOperator(object):

    def select_solution(self, population, metric):
        pass

class SelectionOperatorTournament(SelectionOperator):

    def __init__(self, tournament_size):
        self.tournament_size = tournament_size

    def select_solution(self, population, metric):
        tournament = choices(population, k=self.tournament_size)
        if metric.greater_is_better:
            winner = max(tournament, key=attrgetter('value'))
        else:
            winner = min(tournament, key=attrgetter('value'))
        if type(winner) is Solution: return winner
        else: return sample(set(winner), 1)

