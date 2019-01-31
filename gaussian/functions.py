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
            cov_matrices.append(np.cov(np.random.rand(self.dimension, 600)))
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
        # assert(sum(params) == 1.0)
        self.weights = params
        return params

    def initialize_gamma(self):
        k = np.random.rand(self.no_of_points, self.no_of_gaussians)
        sums = np.sum(k, axis=0)
        for i in xrange(self.no_of_gaussians):
            k[:, i] = k[:, i]/sums[i]
        self.gamma = k
        return k

    def update(self):
        '''
        gamma rowise is point wise and column wise is gaussian wise
        '''
        gamma = np.ones((self.no_of_points, self.no_of_gaussians)) 
        for i in xrange(self.no_of_points):
            l = 0.0
            for j in xrange(self.no_of_gaussians):
                prob = multivariate_normal.pdf(self.points[i], self.means[j], self.cov_matrices[j])*self.weights[j]
                # print 'a'
                gamma[i, j] = prob
                l += prob
            for j in xrange(self.no_of_gaussians):
                gamma[i, j] = gamma[i, j]/l
        self.gamma = gamma
        s = np.sum(gamma, axis=0) #row wise sum of gamma matrix
        for i in xrange(self.no_of_gaussians):
            self.weights[i] = s[i]/self.no_of_points #updating weights
            mean = np.zeros((self.dimension, ))
            for j in xrange(self.no_of_points): 
                print np.argwhere(np.isnan(gamma[j, i]))
                k = np.multiply(self.points[j], gamma[j, i])
                mean += k
            self.means[i] = mean/s[i] #updating means
        for i in xrange(self.no_of_gaussians):
            g = np.array(self.points, (self.no_of_points, self.dimension))-np.reshape(self.means[i], (1, self.dimension))        
            self.cov_matrices[i] = np.dot(g.T, self.gamma[:, i]*g) #updating covariance matrices
        print self.means


    def update_inverse(self):
        s = np.sum(self.gamma, axis=0)
        self.weights = list()
        self.means = list()
        self.cov_matrices = list()
        for i in xrange(self.no_of_gaussians):
            self.weights.append(0)
            self.means.append(0)
            self.cov_matrices.append(0)
        for i in xrange(self.no_of_gaussians):
            self.weights[i] = s[i]/self.no_of_points
            mean = np.zeros((self.dimension, ))
            for j in xrange(self.no_of_points): 
                k = np.multiply(self.points[j], self.gamma[j, i])
                mean += k
            self.means[i] = mean/s[i]
        for i in xrange(self.no_of_gaussians):
            k = np.array(self.points)
            g = np.reshape(k, (self.no_of_points, self.dimension))-np.reshape(self.means[i], (1, self.dimension))        
            self.cov_matrices[i] = np.dot(g.T, np.reshape(self.gamma[:, i], (self.gamma.shape[0], 1))*g)
        gamma = np.ones((self.no_of_points, self.no_of_gaussians)) 
        for i in xrange(self.no_of_points):
            l = 0.0
            for j in xrange(self.no_of_gaussians):
                prob = multivariate_normal.pdf(self.points[i], self.means[j], self.cov_matrices[j], allow_singular=True)*self.weights[j]
                # print 'a'
                gamma[i, j] = prob
                l += prob
            for j in xrange(self.no_of_gaussians):
                gamma[i, j] = gamma[i, j]/l
        self.gamma = gamma
        


    def iterate(self):
        for i in xrange(self.no_of_iterations):
            if  i == 0:
                if self.means == None:
                    self.initilize_means()
                if self.cov_matrices == None:
                    self.initilize_cov_matrices()
                if self.weights == None:
                    self.initialize_parameters()
            print 'iteration - '+str(i+1)
            self.update()
            print ''
            print 'iteration complete'

    def iterate_inverse(self):    
        for i in xrange(self.no_of_iterations):
            if i == 0:
                if self.gamma == None:
                    self.initialize_gamma()
            print 'iteration - '+str(i+1)
            self.update_inverse()
            print ''
        print '#####Iterations complete#######'

    def log_likelihood(self):
        return 1


