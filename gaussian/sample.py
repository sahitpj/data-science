import numpy as np 
import random, math


class GaussianMMSampler(object):
    def __init__(self, weights , mean, cov_matrices):
        self.weights = weights
        self.means = mean
        self.cov_matrices = cov_matrices
        self.no_of_gaussians = len(weights)


    def mixture_sampling(self, no_of_points):
        points = list()
        gaussian_id = list()
        for i in xrange(self.no_of_gaussians):
            gs = GaussianSampler(self.means[i], self.cov_matrices[i])
            sample_points = gs.sample_list(int(math.ceil(self.weights[i]*no_of_points)))
            points.extend(sample_points)
            for j in xrange(int(math.ceil(self.weights[i]*no_of_points))):
                gaussian_id.append(i+1)
        assert(len(points) >= no_of_points)
        return points, gaussian_id



class GaussianSampler(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self):
        return np.random.multivariate_normal(self.mean, self.cov)

    def sample_list(self, no_of_points):
        sample_points = list()
        for i in xrange(no_of_points):
            point = np.random.multivariate_normal(self.mean, self.cov)
            sample_points.append(point)
        return sample_points #being returnd as list of points.