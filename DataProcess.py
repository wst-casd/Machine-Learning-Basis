
import numpy as np


def readData( fileName, delim=' ' ):
	data = []
	labels = []

	for str in open( fileName ).readlines( ):
		lineString = str.strip().split( delim )
		data.append( [ float(elem) for elem in lineString[0:len(lineString)-1] ] )
		labels.append( int( lineString[-1] ) )

	return np.array( data ), np.array( labels )

# Mean Normalize
def meanNormalize( dataSet ):
	
	mu = np.mean(dataSet, axis=0)		# Calculate the mean of each feature
	sigma = np.std(dataSet, axis=0)		# Calculate the Standard deviation of each feature

	dataSize = dataSet.shape[0]

	dataset_norm = (dataSet - np.tile(mu, (dataSize, 1))) / np.tile(sigma, (dataSize, 1))

	return dataset_norm, mu, sigma

# mixmin Normalize
def maxminNormalize( dataSet ):
	
	minVals = np.min(dataSet, axis=0)
	maxVals = np.max(dataSet, axis=0)
	ranges = maxVals - minVals + 1e-10

	dataSize = dataSet.shape[0]

	dataSet_norm = (dataSet - np.tile(minVals, (dataSize, 1))) / np.tile(ranges, (dataSize, 1))

	return dataset_norm, minVals, ranges 






