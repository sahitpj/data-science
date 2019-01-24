import numpy as np

class EstimationMaximisation(object):
    def __init__(self, points, no_of_iterations, no_of_gaussians, parametric="Yes"):
        self.points = points #list of numpy arrays
        self.no_of_iterations = no_of_iterations
        self.parametric = parametric
        self.no_of_gaussians = no_of_gaussians
        self.dimension = points[0].shape[0]
        self.means = None
        self.cov_matrices = None
        self.weights = None

    def initilize_means(self):
        means = list()
        for i in xrange(self.no_of_gaussians):
            means.append(np.random.rand(self.dimension, ))
        return means


    def initilize_cov_matrices(self):


    def initialize_parameters(self):


