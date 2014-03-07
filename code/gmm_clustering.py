"""
Author: 
Contact: 
Date: March 5, 2014

This implements GMM (Gausian Mixture Model) clustering.

Usage: gmm_clustering <master> <input_file> <#clusters> <output_file> <covergence_order (optional)> <seed (optional)>
To run this, use something like:
    ../../bin/pyspark gmm_clustering.py local kmeans_data.txt 2 output.txt 6 1
    (or use ./run)

For more info, please refer to README.txt
"""

import sys
import os
import numpy as np
from pyspark import SparkContext
# The following bit is for writing RDD into a file
import tempfile
from fileinput import input
from glob import glob

# TODO: In Scipy 0.14.0, multivariate_normal is available, but not in the current version (0.13).
from _multivariate import multivariate_normal
# So when it's released, do the following instead.
# from scipy.stats import multivariate_normal 

# input example: "1.2, 3.2, 0.5" (string)
# output example: [1.2, 3.2, 0.5] (numpy array)
def parseVector(line):
    return np.array([float(x) for x in line.split(',')])

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

## The following two methods are for writing out the output
# input: one-dimensional numpy array ([1,2,3])
# output: csv-like string ("1,2,3")
def np_array_to_csv(np_array):
    return str(np_array.tolist())[1:-1]

# input: parameters and a file path
# output: outputs the resulst (parameters) in the specified file
def output_results(pi, centers, cov_matrices, file_path):
   f = open(file_path, 'w+') 
   f.write(np_array_to_csv(pi) + "\n")
   for i in range(0, len(pi)):
       f.write(np_array_to_csv(centers[i]) + "\n")
       for j in range(0, len(cov_matrices[i])):
           f.write(np_array_to_csv(cov_matrices[i][j]) + "\n")
   f.close()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print >> sys.stderr, "Usage: gmm_clustering <master> <input_file> <#clusters> <output_file> <covergence_order (optional)> <seed (optional)>"
        exit(-1)

    ## Read the input from the text file
    sc = SparkContext(sys.argv[1], "PythonGMMClustering")
    lines = sc.textFile(sys.argv[2])
    data_points = lines.map(parseVector).cache()
    NUM_CLUSTERS = int(sys.argv[3])
    OUTPUT_FILE = (sys.argv[4])
    CONVERGENCE_ORDER = 6 if len(sys.argv) < 6 else int(sys.argv[5])
    # Covergence condition: (decrease in log probability) <= (initial decrease in log probability) * 10^(-CONVERGENCE_ORDER)
    SEED = 1 if len(sys.argv) < 7 else int(sys.argv[6])
    num_points = data_points.count()
    print("Number of points we have in the file:", num_points)

    ## Initialize parameters
    centers = data_points.takeSample(False, NUM_CLUSTERS, SEED) # randomly choose points to initialize the centers, withReplacement = False
    dim = len(centers[0]) # dim = number of features
    cov_matrices = [ np.identity(dim) ] * NUM_CLUSTERS # choose the identity matrices as the initial covariance matrices
    pi = np.ones(NUM_CLUSTERS) / NUM_CLUSTERS # choose the uniform distribution as the initial pi

    loop_count = 0 # counts the number of iterations we execute
    # Initial log probability
    log_prob = data_points \
        .map(lambda p: np.log( sum([pi[k] * multivariate_normal(centers[k], cov_matrices[k]).pdf(p) for k in range(0, NUM_CLUSTERS)]) ) ) \
        .reduce(lambda a, b: a + b)

    # Until convergence
    while True:
        ## E Step: Compute the responsibilities
        # NOTE: cache resp because we are going to use it a couple of times below, and also because we want to fix the values so they don't change later.
        resp = data_points.map(lambda p: (p, responsibilities(p, pi, centers, cov_matrices) ) ).cache()
        # format of resp: (point, numpy_array_of_responsibilities)
            
        ## M Step: Update the parameters
        # In the following part, (p, r) stands for (point, responsibility)
        pi = resp.map(lambda (p, r): r).reduce( lambda r1, r2: r1 + r2 ) / num_points
        centers = resp.map(lambda (p, r): np.array([p * k for k in r])) \
            .reduce(lambda t1, t2: t1 + t2) / num_points / np.array([[i] * dim for i in pi]) # fit the shape of the array so we can do division
        temp1 = np.array([transpose_and_multiply(c) for c in centers ])
        temp2 = resp.map(lambda (p, r): np.array([k * transpose_and_multiply(p) for k in r])) \
            .reduce(lambda t1, t2: t1 + t2) / num_points / np.array([[i] * dim * dim for i in pi]).reshape(NUM_CLUSTERS, dim, dim) # fit the shape of the array so we can do division
        cov_matrices = temp2 - temp1

        ## Update log probability
        old_log_prob = log_prob
        log_prob = data_points \
            .map(lambda p: np.log( sum([pi[k] * multivariate_normal(centers[k], cov_matrices[k]).pdf(p) for k in range(0, NUM_CLUSTERS)]) ) ) \
            .reduce(lambda a, b: a + b)
        
        ## At the first loop, set the initial decrease
        if loop_count == 0:
            initial_decrease_in_log_prob = log_prob - old_log_prob
        decrease_in_log_prob = log_prob - old_log_prob
        loop_count += 1 # increase the loop before break: indicates the number of iterations DONE.

        ## Check for covergence: decrease in log probability is very very small
        if decrease_in_log_prob <= initial_decrease_in_log_prob * 10**(-CONVERGENCE_ORDER):
            break
       
    ## Output the results (parameters)
    output_results(pi, centers, cov_matrices, OUTPUT_FILE)

    ## The following part is for writing individual points and their hard assignments in to the file -- referred to Alim's code to understand the procedure.
    # Thanks, Alim!
    # create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    temp_file.close()
    # Save the point and the most probable cluster that correspond to it
    # Hack: string[0:-2] gets rid of ".0" in "1.0, 1.2, 1.0"
    # We want to get rid of it because it's just an index, so it's supposed to be an integer
    resp.map(lambda (p, r): np_array_to_csv( np.append(p, np.argmax(r)) )[0:-2]).saveAsTextFile(temp_file.name)
    # get the temp file using glob (gets a list of files)
    source_files = glob(temp_file.name + "/part-0000*")
    # append the the source files to the output file 64KB at a time
    file = open(OUTPUT_FILE,'a')
    for f in source_files:
        source_file = open(f,'r')
        while True:
             input_data = source_file.read(65536)
             if input_data:
                 file.write(input_data)
             else:
                break

    ## Print out the results for immediate inspection
    print("loop_count", loop_count)
    print("centers", centers)
    print("cov_matrices", cov_matrices)
    print("pi", pi) 
