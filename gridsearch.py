from pprint import pprint

from Detector import Detector

__author__ = 'Rik Smit (h.smit.6@student.rug.nl)'

projectDirectory = '/home/rik-target/owncloud-rixmit/Studie/Droneproject/dataset/cows/'
datasetIds = [
    'DJI_0005_cut_233-244',
    'DJI_0007_cut_22-65',
    'DJI_0081'
]

detector = Detector(projectDirectory, datasetIds)
detector.featureType = 'HSV-HIST'
detector.classifierType = 'SVM-RBF'

datasetIdx = 0  # used for interset (which set is tested on

parameterOptions = { # lists of different classifier parameters values that should be tested
    'C': [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256],
    'gamma': [0.1, 1, 10, 100, 1000],
    'n_neighbors': [1, 2, 3, 4, 5, 10]
}


# singleset test
detector.gridSearchSingleSet(parameters=parameterOptions, datasetIdx=datasetIdx)

# crossset test
detector.gridSearch(parameters=parameterOptions)
