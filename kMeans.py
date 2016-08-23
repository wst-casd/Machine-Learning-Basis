
import numpy as np

def distEuclidean( A, vecB ):
	return np.sqrt( np.sum( (A - vecB)**2, axis = 1 ) )


# This function initializes K centroids which will be used in K-Means on the dataset data
def initCentroids( data, K ):
	

	return centroids


# findClosestCenroids computes the centroid memberships for every example
def findClosestCenroids( data, centroids ):
	m = data.shape[0]
	clusterAssment = np.array( np.zeros( (m, 2) ) )
	clusterChanged = False

	for i in range( m ):
		distances = distEuclidean( centroids, data[i, :] )
		minDistIdx = np.argmin( distances )				# minDistIdx indicates which centroid the example belongs to
		minDist = distances[minDistIdx]
		
		if( clusterAssment[i, :] != minDistIdx ):
			clusterChanged = True

		clusterAssment[i, :] = minDistIdx, minDist**2

	return clusterAssment, clusterChanged


# returs the new centroids by computing the means of the data points assigned to each centroid
def computeCentroids( data, idx, K ):
	n = data.shape[1]
	centroids = np.array( np.zeros( (K, n) ) )

	for i in range( K ):
		ptsCluster = data[ np.nonzero( idx == i )[0], : ]
		centroids[i, :] = np.mean( ptsCluster, axis=0 )

	return centroids


def kMeans( data, initial_centroids ):
	K = initial_centroids.shape[0]
	centroids = initial_centroids
	clusterChanged = True

	while clusterChanged:
		clusterChanged = False
		clusterAssment, clusterChanged = findClosestCenroids( data, centroids )
		centroids = computeCentroids( data, clusterAssment[:, 0], K )

	return centroids, clusterAssment