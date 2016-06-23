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

runs = 20
for active in [True, False]:
    print("Testing (active=%s)" % active)
    results = None
    for run in range(runs):
        result = detector.activeLearningTest(active, datasetIdx=1) # use dataset DJI_0007_cut_22-65 as it is big enough
        if not results:
            results = result
        else:
            results = [x + y for x, y in zip(results, result)]

    pprint([x / runs for x in results])


