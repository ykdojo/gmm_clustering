# TODO: Update this
"""
This implements GMM (Gausian Mixture Model) clustering.

To run this, use something like:
    ../../bin/pyspark gmm_clustering.py local kmeans_data.txt 2 0.1
"""

import sys
import numpy as np
import scipy as sp # for multivariate normal
from pyspark import SparkContext

# TODO: In Scipy 0.14.0, multivariate_normal is available, but not in the current version.
from _multivariate import multivariate_normal
# So when it's released, do the following instead.
# from scipy.stats import multivariate_normal 

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

# input: point
# output: (point, numpy_array_of_responsibilities)
def responsibilities(point, pi, centers, cov_matrices):
    r = np.zeros(len(pi))
    for k in range(0, len(pi)):
        r[k] = pi[k] * multivariate_normal(centers[k], cov_matrices[k]).pdf(point)
    return r / sum(r)

# input: numpy_array
# output: numpy_array * numpy_array_transposed (in matrix)
def transpose_and_multiply(np_array):
    # need to cast to matrix in order to tranpose a one-dimentional array
    return np.transpose(np.matrix(np_array)).dot(np.matrix(np_array))

if __name__ == "__main__":
    # TODO: edit the following part later
    if len(sys.argv) < 4:
        print >> sys.stderr, "Usage: gmm_clustering <master> <input_file> <#clusters>"
        exit(-1)

    # Read the input from the text file
    sc = SparkContext(sys.argv[1], "PythonGMMClustering")
    lines = sc.textFile(sys.argv[2])
    data_points = lines.map(parseVector).cache()
    num_clusters = int(sys.argv[3])
    num_points = data_points.count()
    print("Number of points we have in the file:", num_points)

    ## Initialize parameters
    centers = data_points.take(num_clusters) # randomly choose points to initialize the centers
    dim = len(centers[0]) # dim = number of features
    cov_matrices = [ np.identity(dim) ] * num_clusters
    pi = np.ones(num_clusters) / num_clusters

    ## E Step: Compute the responsibilities
    resp = data_points.map(lambda p: (p, responsibilities(p, pi, centers, cov_matrices) ) )
    # format of resp: (point, numpy_array_of_responsibilities)
        
    ## M Step: Update the parameters
    # In the following part, (p1, r1) stands for (point_1, responsibility_1)
    pi = resp.reduce(lambda (p1, r1), (p2, r2) : r1 + r2) / num_points  # we don't need points here, so ignore them (just look at responsibilities)
    temp = resp.map(lambda (p, r): np.array([p * k for k in r])) # make a 2-d array
    centers = temp.reduce(lambda t1, t2: t1 + t2) / num_points / np.array([[i] * dim for i in pi]) # fit the shape of the array so we can do division
    temp2 = np.array([transpose_and_multiply(c) for c in centers ])
    
    temp3 = resp.map(lambda (p, r): np.array([k * transpose_and_multiply(p) for k in r]))
    temp3 = temp3.reduce(lambda t1, t2: t1 + t2) / num_points / np.array([[i] * dim * dim for i in pi]).reshape(num_clusters, dim, dim) # fit the shape of the array so we can do division
    cov_matrices = temp3 - temp2
    import pdb; pdb.set_trace()
    
    # Check for convergence

    # Print out the result

#    print("centers", centers)
#    print("cov_matrices", cov_matrices)
#    print("pi", pi) 
