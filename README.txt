========================
Author: Yosuke Sugishita
Contact: yosuke.sugishita@alumni.ubc.ca
========================


Hello my peer-reviewer!!


**HOW TO RUN THE CODE
Format: ./run <PYSPARK_FILEPATH> <master> <input_file> <#clusters> <output_file> <covergence_order (optional)> <seed (optional)>
Example: ./run ../../bin/pyspark local data/data.txt 3 data/output.txt 6 1

NOTE the covergence condition: (decrease in log probability) <= (initial decrease in log probability) * 10^(-CONVERGENCE_ORDER)
The default CONVERGENCE_ORDER is 6.


**DIRECTORY STRUCTURE
- run
- README.txt 
- data/ -> data.txt
- code/ -> gmm_clustering.py (main code), _multivariate.py


**PARALLELIZATION STRATEGY
E-Step, M-Step, and calculating log probabilities are all parallelized.

<E-Step>
[data_points] contains all the data points.  Responsibilities are computed for each data point in parallel.

<M-Step>
[resp] contains (point, responsibilities) pairs. pi, centers, and (components of) covariance matrices are computed in parallel.  I used numpy as much as possible here to simplify the code.

<Log probabilities>
The log probability for each point in [data_points] is computed in parallel before being summed together in the reduce operation.

NOTE: I chose to use log probability for this operation so we will be able to try different initial parameters and compare them in the future.  As it's possible that we get stuck in a local maxima, it would be important to be able to do it.


**NOTE ABOUT DELIMITERS
The kmeans data in the spark examples is delimited by spaces and not commas, so I'll follow this convention in my submission as suggested by Brendan.
https://github.com/apache/incubator-spark/blob/master/data/kmeans_data.txt
