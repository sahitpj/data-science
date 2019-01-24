from utils import meta, covariance_data
from gaussian import GaussianMMSampler, GaussianSampler, EstimationMaximisation


no_of_gaussians, dimensions, weights, centers = meta('set1')
cov_matrices = covariance_data(no_of_gaussians, 'set1')
assert(dimensions == cov_matrices[0].shape[0])
assert(len(centers) == len(cov_matrices))

gs1 = GaussianSampler(centers[0], cov_matrices[0])

gmms = GaussianMMSampler(weights, centers, cov_matrices)

points = gs1.sample_list(10)
# points_mm, gaussians_id = gmms.mixture_sampling(1000)


em = EstimationMaximisation(points, 1, 4, "Yes")

m = em.initilize_cov_matrices()
params = em.initialize_parameters()
print params