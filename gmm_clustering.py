# TODO: Update and check this part
"""
Author: Yosuke Sugishita
Contact: yosuke.sugishita@ubc.alumni.ubc.ca

This implements GMM (Gausian Mixture Model) clustering.

Usage: gmm_clustering <master> <input_file> <#clusters> <output_file> <covergence_order (optional)> <seed (optional)>
To run this, use something like:
    ../../bin/pyspark gmm_clustering.py local kmeans_data.txt 2 output.txt 6 1
"""

import sys
import os
import numpy as np
import scipy as sp # for multivariate normal
from pyspark import SparkContext

# TODO: In Scipy 0.14.0, multivariate_normal is available, but not in the current version (0.13).
from _multivariate import multivariate_normal
# So when it's released, do the following instead.
# from scipy.stats import multivariate_normal 

# input example: "1.2, 3.2, 0.5" (string)
# output example: [1.2, 3.2, 0.5] (numpy array)
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
    if len(sys.argv) < 5:
        print >> sys.stderr, "Usage: gmm_clustering <master> <input_file> <#clusters> <output_file> <covergence_order (optional)> <seed (optional)>"
        exit(-1)

    ## Read the input from the text file
    sc = SparkContext(sys.argv[1], "PythonGMMClustering")
    lines = sc.textFile(sys.argv[2])
    data_points = lines.map(parseVector).cache()
    NUM_CLUSTERS = int(sys.argv[3])
    CONVERGENCE_ORDER = 6 if len(sys.argv) < 6 else int(sys.argv[5])
    # Covergence condition: (decrease in log probability) <= (initial decrease in log probability) * 10^(-CONVERGENCE_ORDER)
    SEED = 1 if len(sys.argv) < 7 else int(sys.argv[6])

    num_points = data_points.count()
    print("Number of points we have in the file:", num_points)

    ## Initialize parameters
    # TODO: try this multiple times with different seeds?
    centers = data_points.takeSample(False, NUM_CLUSTERS, SEED) # randomly choose points to initialize the centers, withReplacement = False
    dim = len(centers[0]) # dim = number of features
    cov_matrices = [ np.identity(dim) ] * NUM_CLUSTERS
    pi = np.ones(NUM_CLUSTERS) / NUM_CLUSTERS

    loop_count = 0
    initial_decrease_in_log_prob = 0
    decrease_in_log_prob = np.inf
    # Initial log probability
    log_prob = data_points \
        .map(lambda p: np.log( sum([pi[k] * multivariate_normal(centers[k], cov_matrices[k]).pdf(p) for k in range(0, NUM_CLUSTERS)]) ) ) \
        .reduce(lambda a, b: a + b)

    # Until convergence
    while True:
        ## E Step: Compute the responsibilities
        # NOTE: cache resp because we are going to use it a couple of times below, and also because we want to fix the values.
        resp = data_points.map(lambda p: (p, responsibilities(p, pi, centers, cov_matrices) ) ).cache()
        # format of resp: (point, numpy_array_of_responsibilities)
            
        ## M Step: Update the parameters
        # In the following part, (p1, r1) stands for (point_1, responsibility_1)
        pi = resp.map(lambda (p, r): r).reduce( lambda r1, r2: r1 + r2 ) / num_points
        centers = resp.map(lambda (p, r): np.array([p * k for k in r])) \
            .reduce(lambda t1, t2: t1 + t2) / num_points / np.array([[i] * dim for i in pi]) # fit the shape of the array so we can do division
        temp2 = np.array([transpose_and_multiply(c) for c in centers ])
        temp3 = resp.map(lambda (p, r): np.array([k * transpose_and_multiply(p) for k in r])) \
            .reduce(lambda t1, t2: t1 + t2) / num_points / np.array([[i] * dim * dim for i in pi]).reshape(NUM_CLUSTERS, dim, dim) # fit the shape of the array so we can do division
        cov_matrices = temp3 - temp2

        ## Update log probability
        old_log_prob = log_prob
        log_prob = data_points \
            .map(lambda p: np.log( sum([pi[k] * multivariate_normal(centers[k], cov_matrices[k]).pdf(p) for k in range(0, NUM_CLUSTERS)]) ) ) \
            .reduce(lambda a, b: a + b)
        
#        # just some code for debugging..
#        with open("debug.txt", "a") as myfile:
#            myfile.write(str(log_prob))
#            myfile.write("\n")

        ## At the first loop, set the initial decrease
        if loop_count == 0:
            initial_decrease_in_log_prob = log_prob - old_log_prob
        decrease_in_log_prob = log_prob - old_log_prob
        assert log_prob >= old_log_prob # making sure the log probability decreases every time
        loop_count += 1 # increase the loop before break: indicates the number of iterations DONE.

        ## Check for covergence: decrease in log probability is very very small
        if decrease_in_log_prob <= initial_decrease_in_log_prob * 10**(-CONVERGENCE_ORDER):
            break
       
    # Print out the result
    print("loop_count", loop_count)
    print("centers", centers)
    print("cov_matrices", cov_matrices)
    print("pi", pi) 
