
import numpy as np

def runPCA( priorData, retainRatio = 0.99 ):
	
	m = priorData.shape[0]
	zeroMeanData = priorData - np.mean( priorData, axis=0 )

	U, S, VT = np.linalg.svd( np.dot( zeroMeanData.T, zeroMeanData)/(m-1.0) )

	infoSum = np.sum( S )
	infoRetain = 0

	for i in range( S.shape[0] ):
		infoRetain += S[i]

		if infoRetain/infoSum >= retainRatio:
			transformMat = U[ :, 0:i ]
			break

	lowData = np.dot( priorData, transformMat )

	return lowData, transformMat