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

trainsetIdxs = [0, 1, 2]
testsetIdx = 1
displayOptions = {
    'enabled': True,  # display anything at all
    'frame': True,  # video frame image
    'detections': False,  # detected objects
    'suppressed_detections': True,
    'sliding_window': False,
    'ground_truth': False
}

detector.test_detect(trainsetIdxs, testsetIdx, harvest=True, autoAnnotate=True, displayOptions=displayOptions)
