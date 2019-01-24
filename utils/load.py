import numpy as np

def meta(dataset):
    filepath = 'data/'+dataset+'/centers.txt'
    no_of_gaussians = None
    dimensions = None
    weights = list()
    centers = list()
    f = open(filepath, 'r')
    lines = f.readlines()
    for i in xrange(len(lines)):
        if i == 0:
            lines[i] = map(int, lines[i][:-2].split(','))
            no_of_gaussians, dimensions = lines[0]
        elif i == 1:
            lines[i] = map(float, lines[i][:-3].split(','))
            weights = lines[1]
            assert(sum(weights) == 1.0)
        else:
            centers.append(np.array(map(float, lines[i][:-3].split(','))))
    return no_of_gaussians, dimensions, weights, centers #outputs all data in lists or in values



def covariance_data(no_of_gaussians, dataset):
    cov_matrices = list()
    for i in xrange(no_of_gaussians):
        filepath = 'data/'+dataset+'/cov_'+str(i+1)+'.txt'
        cov = np.loadtxt(filepath, delimiter=',', comments='#')
        cov_matrices.append(cov)
    return cov_matrices #returned as list of numpy matrices