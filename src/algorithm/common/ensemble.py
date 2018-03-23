from numpy import mean, zeros
from copy import deepcopy

class Ensemble(object):
    """
    Class represents ensemble learning technique. In short, ensemble techniques predict output over a meta learner
    that it self is supplied with output of a number of base learners.

    Attributes:
        base_learner: Base learner algorithm that supplies meta learner.
        number_learners: Number of base learners.
        meta_learner: Meta learner that predicts output, based on base learner predictions.
        learners: List, containing the trained base learners.

    Notes:
        base_learner needs to support fit() and predict() function.
        meta_learner function needs to support numpy ndarray as input.
    """

    def __init__(self, base_learner, number_learners, meta_learner=mean):
        self.base_learner = base_learner
        self.number_learners = number_learners
        self.meta_learner = meta_learner
        self.learners = list()

    def fit(self, input_matrix, target_vector, metric, verbose=False):
        """Trains learner to approach target vector, given an input matrix, based on a defined metric."""

        for i in range(self.number_learners):
            if verbose: print(i)
            # Creates deepcopy of base learner.
            learner = deepcopy(self.base_learner)
            # Trains base learner.
            learner.fit(input_matrix, target_vector, metric)
            # Adds base learner to list.
            self.learners.append(learner)

    def predict(self, input_matrix):
        """Predicts target vector, given input_matrix, based on trained ensemble."""

        # Creates prediction matrix.
        predictions = zeros([input_matrix.shape[0], self.number_learners])
        # Supplies prediction matrix with predictions of base learners.
        for learner, i in zip(self.learners, range(len(self.learners))):
            predictions[:, i] = learner.predict(input_matrix)
        # Applies meta learner to prediction matrix.
        return self.meta_learner(predictions, axis=1)
