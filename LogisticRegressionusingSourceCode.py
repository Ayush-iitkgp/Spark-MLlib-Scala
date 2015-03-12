
from collections import namedtuple
from math import exp
from os.path import realpath
import sys

import numpy as np
from pyspark import SparkContext


D = 16  # Number of dimensions


# Read a batch of points from the input file into a NumPy matrix object. We operate on batches to
# make further computations faster.
# The data file contains lines of the form <label> <x1> <x2> ... <xD>. We load each block of these
# into a NumPy array of size numLines * (D + 1) and pull out column 0 vs the others in gradient().
def readPointBatch(iterator):
    strs = list(iterator)
    matrix = np.zeros((len(strs), D + 1))
    for i in xrange(len(strs)):
        matrix[i] = np.fromstring(strs[i].replace(',', ' '), dtype=np.float32, sep=' ')
    return [matrix]

#if __name__ == "__main__":
 #   if len(sys.argv) != 3:
  #      print >> sys.stderr, "Usage: logistic_regression <file> <iterations>"
   #     exit(-1)

    sc = SparkContext(appName="PythonLR")
    points = sc.textFile(1 2.857738033247042 0 2.061393766919624 2.619965104088255 0 0 2.000347299268466 0 2.228387042742021 2.228387042742023 0 2.055002875864414 0 0 0 0
).mapPartitions(readPointBatch).cache()
    iterations = 100

    # Initialize w to a random value
    w = 2 * np.random.ranf(size=D) - 1
    print "Initial w: " + str(w)

    # Compute logistic regression gradient for a matrix of data points
    def gradient(matrix, w):
        Y = matrix[:, 0]    # point labels (first column of input file)
        X = matrix[:, 1:]   # point coordinates
        # For each point (x, y), compute gradient function, then sum these up
        return ((1.0 / (1.0 + np.exp(-Y * X.dot(w))) - 1.0) * Y * X.T).sum(1)

    def add(x, y):
        x += y
        return x

    for i in range(iterations):
        print "On iteration %i" % (i + 1)
        w -= points.map(lambda m: gradient(m, w)).reduce(add)

    print "Final w: " + str(w)
