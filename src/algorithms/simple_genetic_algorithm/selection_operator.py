from random import choices, sample
from algorithms.simple_genetic_algorithm.solution import Solution
from operator import attrgetter

class SelectionOperator(object):

    def select_solution(self, population):
        pass

class SelectionOperatorTournament(SelectionOperator):

    def __init__(self, tournament_size):
        self.tournament_size = tournament_size

    def select_solution(self, population):
        tournament = choices(population, k=self.tournament_size)
        winner = min(tournament, key=attrgetter('mean_error'))
        if type(winner) is Solution: return winner
        else: return sample(set(winner), 1)

