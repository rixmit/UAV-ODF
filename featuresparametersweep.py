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


testtype = 'interset'  # or crossset
parameterOptions = { # lists of different parameter values that should be tested for the feature descriptors
    'channelsSet': [[0, 1, 2], [1], [0, 1]],
    'histSizeSet': [32, 8],
    'orientationsSet': [1, 2, 4, 8],
    'ppcSet': [8, 16, 32],
    'cpbSet': [2, 4, 8]
}
datasetIdx = 0  # used for interset (which set is tested on

classifierParameters = { # single paramters that are used for the classifier
    'C': [1],
    'gamma': [0.1],
    'n_neighbors': [5]
}

detector.parameterSweep(testtype=testtype, parametersOptions=parameterOptions, datasetIdx=datasetIdx, classifierParameters=classifierParameters)


