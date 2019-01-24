import numpy as np
import random
from scipy.stats import multivariate_normal

class EstimationMaximisation(object):
    def __init__(self, points, no_of_iterations, no_of_gaussians, parametric="Yes"):
        self.points = points #list of numpy arrays
        self.no_of_points = len(self.points)
        self.no_of_iterations = no_of_iterations
        self.parametric = parametric
        self.no_of_gaussians = no_of_gaussians
        self.dimension = points[0].shape[0]
        self.means = None
        self.cov_matrices = None
        self.weights = None
        self.params = None
        self.gamma = None

    def initilize_means(self):
        means = list()
        for i in xrange(self.no_of_gaussians):
            means.append(np.random.rand(self.dimension, ))
        self.means = means
        return means


    def initilize_cov_matrices(self):
        cov_matrices = list()
        for i in xrange(self.no_of_gaussians):
            cov_matrices.append(np.cov(np.random.rand(self.dimension, 50)))
        self.cov_matrices = cov_matrices
        assert(cov_matrices[0].shape[0] == self.dimension)
        assert(cov_matrices[0].shape[1] == self.dimension)
        return cov_matrices

    def initialize_parameters(self):
        params = list()
        for i in xrange(self.no_of_gaussians):
            params.append(random.uniform(0, 1))
        l = sum(params)
        for i in xrange(self.no_of_gaussians):
            params[i] = params[i]/l
        assert(sum(params) == 1.0)
        self.params = params
        return params


    def update(self):
        '''
        gamma rowise is point wise and column wise is gaussian wise
        '''
        gamma = np.ones((self.no_of_points, self.no_of_gaussians)) 
        for i in xrange(self.no_of_points):
            l = 0.0
            for j in xrange(self.no_of_gaussians):
                prob = multivariate_normal(self.points[i], mean=self.means[j], cov=self.cov_matrices[j])*self.weights[j]
                gamma[i, j] = prob
                l += prob
            for j in xrange(self.no_of_gaussians):
                gamma[i, j] = gamma[i, j]/l
        self.gamma = gamma
        s = np.sum(gamma, axis=0) #row wise sum of gamma matrix
        for i in xrange(self.no_of_gaussians):
            self.weights[i] = s[i]/self.no_of_points
            mean = np.array((self.dimension, ))
            for j in xrange(self.no_of_points):
                t += np.multiply(self.points[j], gamma[i, j])
            self.means[i] = mean/s[i]
        for i in xrange(self.no_of_gaussians):
            g = np.array(self.points, (self.no_of_points, self.dimension))-np.array(self.means[i], (1, self.dimension))        
            self.cov_matrices[i] = np.dot(g.T, gamma[:, i]*g)
        

    def iterate(self):
        for i in xrange(self.no_of_iterations):
            print 'iteration - '+str(i+1)
            self.update
            print ''
            print 'iteration complete'

    def log_likelihood(self):
        return 1

