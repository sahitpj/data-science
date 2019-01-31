from utils import meta, covariance_data
from gaussian import GaussianMMSampler, GaussianSampler, EstimationMaximisation, GaussianRSampler
import math
import numpy as np
from scipy.stats import multivariate_normal


no_of_gaussians, dimensions, weights, centers = meta('set1')
cov_matrices = covariance_data(no_of_gaussians, 'set1')
assert(dimensions == cov_matrices[0].shape[0])
assert(len(centers) == len(cov_matrices))

gs1 = GaussianSampler(centers[0], cov_matrices[0])

gmms = GaussianMMSampler(weights, centers, cov_matrices)

points = gs1.sample_list(10)
# print points[0]
# points_mm, gaussians_id = gmms.mixture_sampling(1000)
em = EstimationMaximisation(points, 2, 2, "Yes")

m = em.initilize_cov_matrices()
params = em.initialize_parameters()
covms = em.initilize_cov_matrices()
means = em.initilize_means()

em.update()


# def normal_pdf(x, mean, cov):
#     dimensions = x.shape[0]/2
#     cov = cov*-40
#     # print (2*math.pi)**dimensions, np.linalg.det(cov)
#     f = 1.0/((2*math.pi)**dimensions)*(np.linalg.det(cov)**0.5)
#     g = 0.5* np.dot(np.dot((x-mean).T, np.linalg.inv(cov)),  (x-mean))
#     # print f, g
#     return f*math.exp(-g)


# p = points[0]

# # print normal_pdf(p, means[1], covms[0])
# print covms[0]
# print  multivariate_normal.pdf(p, means[1], covms[0])

#
# em.iterate()

# x = []
# y = []
# for  num in xrange(10, 100, 10):
#     d = num
#     samples_count = 100000
#     ers = GaussianRSampler(d)
#     # k = ers.sample_list(samples_count)
#     # print len(k)

#     count = 0.0
#     a = (d**0.5)-(d**0.1)
#     b = (d**0.5)+(d**0.1)
#     print a,b
#     for i in xrange(samples_count):
#         l = ers.sample_distance() 
#         # print l
#         if l > a and l < b:
#             count += 1
#     y.append(count/samples_count)
#     x.append(num)
#     print count/samples_count
# plt.scatter(x,y)
# plt.show()
