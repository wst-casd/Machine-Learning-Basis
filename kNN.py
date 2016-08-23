
import numpy as np


def findkNN( testset, dataset, k ):
	# this function can be used to find the k nearest neighbor in the dateset 
    #                              of each sample in testset

	datasetSize = dataset.shape[0]

	kNNIndicies = None

	for x in testset:
		# Calculate the distance between x and each sample in dataset
		distances = np.sum( ( np.tile(x, (datasetSize, 1))-dataset )**2, axis = 1 )

		if kNNIndicies is None:
			kNNIndicies = np.argsort(distances)[0:k]
		else:
			kNNIndicies = np.vstack((kNNIndicies, np.argsort(distances)[0:k]))
	
	kNNIndicies.shape = (-1, k)	  # to make sure that kNNIndicies is a matrix

	return kNNIndicies


def prediction( kNNIndicies, datasetLabels, k ):
	# this function can be used to predict the labels of each sample in testset
	testsetlabels = [];

	for IIndex in kNNIndicies:
		classCount = {}

		for i in range(k):
			voteILabel = datasetLabels[ IIndex[i] ]
			classCount[voteILabel] = classCount.get(voteILabel,0) + 1

		classCount = {value:key for key,value in classCount.items()}
		testsetlabels.append(classCount[max(classCount.keys())])

	return testsetlabels
