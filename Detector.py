import os
import sys
import glob
from pprint import pprint
from random import shuffle
import time
import math
import itertools
import logging

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import PredefinedSplit
from sklearn.grid_search import GridSearchCV
# from boost.mpi import Exception
from skimage.feature import hog
from sklearn import svm
import cv2

from Video import Video
from Suppressor import Suppressor

__author__ = 'Rik Smit (h.smit.6@student.rug.nl)'

class Detector():
    """
        Detect objects in video frames.
    """

    def __init__(self):
        logLevel = logging.DEBUG
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logLevel)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.log.debug('Creating Detector')

        self.projectDirectory = '/home/rik-target/dataset/cows'

        self.datasetIds = [
            'DJI_0005_cut_233-244',
            'DJI_0007_cut_22-65',
            'DJI_0081'
        ]

        # self.classifierType = 'KNN'
        self.classifierType = 'SVM-RBF'
        # self.classifierType = 'SVM-LIN'

        # self.featureType = "HOG"
        self.featureType = "HOG_HSV-HIST"
        # self.featureType = "HSV-HIST"
        self.featureOptions = {
            "HSV-HIST": {
                "channels": [1],
                "mask": None,
                "histSize": [32],
                "ranges": [0, 256]
            },
            "HOG": {
                "winSize": (100, 100),
                "orientations": 4,
                "pixels_per_cell": (32, 32),
                "cells_per_block": (2, 2)
            },
            "HOG_HSV-HIST": {
                "channels": [1],
                "mask": None,
                "histSize": [32],
                "ranges": [0, 256],
                "winSize": (100, 100),
                "orientations": 4,
                "pixels_per_cell": (32, 32),
                "cells_per_block": (3, 3)
            }
        }

    def featuresetHOG_HSVHIST(self):
        """
        Get a several combinations of parameter sets that can be used for grid search
        Parameters are used for the combined HOG/Color histogram feature descriptor
        :return:
        """
        channelsSet = [[0, 1, 2], [1]]
        histSizeSet = [32, 8]
        orientationsSet = [4]
        ppcSet = [32]
        cpbSet = [2, 3]

        parametersets = []
        for channels in channelsSet:
            for histSize in histSizeSet:
                for orientations in orientationsSet:
                    for ppc in ppcSet:
                        for cpb in cpbSet:
                            parametersets.append({
                                "channels": channels,
                                "mask": None,
                                "histSize": [histSize] * len(channels),
                                "ranges": [0, 255] * len(channels),
                                "winSize": (100, 100),
                                "orientations": orientations,
                                "pixels_per_cell": (ppc, ppc),
                                "cells_per_block": (cpb, cpb)
                            })

        return parametersets

    def featuresetColHist(self):
        """
        Get a several combinations of parameter sets that can be used for grid search
        Parameters are used for the Color Histogram feature descriptor
        :return:
        """
        channelsSet = [[0, 1, 2], [0, 1], [0], [1]]
        histSizeSet = [32, 16, 8]

        parametersets = []
        for channels in channelsSet:
            for histSize in histSizeSet:
                parametersets.append({
                    "channels": channels,
                    "mask": None,
                    "histSize": [histSize] * len(channels),
                    "ranges": [0, 255] * len(channels)
                })

        return parametersets

    def featuresetHOG(self):
        """
        Get a several combinations of parameter sets that can be used for grid search
        Parameters are used for the HOG feature descriptor
        :return:
        """
        orientationsSet = [4, 8, 16]
        ppcSet = [8, 16, 32]
        cpbSet = [1, 2, 3]

        parametersets = []
        for orientations in orientationsSet:
            for ppc in ppcSet:
                for cpb in cpbSet:
                    parametersets.append(
                        {
                            "winSize": (100, 100),
                            "orientations": orientations,
                            "pixels_per_cell": (ppc, ppc),
                            "cells_per_block": (cpb, cpb)
                        }
                    )

        return parametersets


    def parameterSweep(self):
        featureType = self.featureType

        self.log.info("Performing parameter sweep on feature type %s" % featureType)

        if featureType == 'HSV-HIST':
            parametersets = self.featuresetColHist()
        elif featureType == 'HOG':
            parametersets = self.featuresetHOG()
        elif featureType == "HOG_HSV-HIST":
            parametersets = self.featuresetHOG_HSVHIST()
        else:
            raise Exception("Unknown feature type: %s" % featureType)

        for parameterset in parametersets:
            self.featureOptions[featureType] = parameterset
            # bestScore = self.gridSearchSingleSet()
            bestScore = self.gridSearch()
            print(bestScore)

    def getObjectsIndices(self, objectsDirectory):
        """
        Get all the unique indices of the objects in the directory.
        Each object (e.g. a Cow) has a unique index
        :param objectsDirectory:
        :return:
        """
        objectsIndices = set()
        files = os.listdir(objectsDirectory)
        for file in files:
            parts = file.split('_')
            objectsIndices.add(int(parts[1]))

        return tuple(objectsIndices)

    def gridSearch(self):
        """
        Perform a grid search using all the available datasets
        Goals is to find the optimal classifier parameters

        :return:
        """

        datasets = []
        for setIdx, datasetId in enumerate(self.datasetIds):
            dataset = {
                'id': datasetId,
                'videoFile': os.path.join(self.projectDirectory, 'videos', datasetId+".MOV"),
                'cutouts': {
                    'posDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'pos'),
                    'negDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'neg')
                },
                'labelsFile': os.path.join(self.projectDirectory, 'labels', datasetId+"_output.txt"),
                'framesDir': os.path.join(self.projectDirectory, 'frames', datasetId),
            }

            posCutoutFiles = (glob.glob(dataset['cutouts']['posDir'] + "/*.png"))
            posLabels = [True] * len(posCutoutFiles)
            negCutoutFiles = (glob.glob(dataset['cutouts']['negDir'] + "/*.png"))
            negLabels = [False] * len(negCutoutFiles)
            samples = []
            for cutoutFile, label in zip(posCutoutFiles + negCutoutFiles, posLabels + negLabels):
                samples.append({
                    'fileName': cutoutFile,
                    'label': label,
                    'features': self.getImageFeatures(cv2.imread(cutoutFile)),
                    'foldIdx': setIdx
                })

            dataset['samples'] = samples

            datasets.append(dataset)

        X = []
        y = []
        test_fold = []
        for dataset in datasets:
            X.extend([sample['features'] for sample in dataset['samples']])
            y.extend([sample['label'] for sample in dataset['samples']])
            test_fold.extend([sample['foldIdx'] for sample in dataset['samples']])

        ps = PredefinedSplit(test_fold=test_fold)

        # est = svm.SVC()
        # parameters = {'kernel':['rbf'], 'C':[1, 2, 3, 5], 'gamma': [0.1, 1, 10, 100, 1000]}
        # parameters = {'kernel':['rbf'], 'C':[1, 3, 10]}

        est = svm.SVC()
        parameters = {'kernel': ['linear'], 'C': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)}
        # parameters = {'kernel': ['linear'], 'C': (1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256)}
        # est = KNeighborsClassifier()
        # parameters = {'n_neighbors':[1, 2, 3, 4, 5, 10]}

        clf = GridSearchCV(
            estimator=est,
            param_grid=parameters,
            n_jobs=1,
            pre_dispatch='2*n_jobs',
            iid=False,
            refit=True,
            cv=ps
        )
        clf.fit(X, y)
        pprint(clf.grid_scores_)
        pprint(clf.best_score_)
        pprint(clf.best_params_)

        for grid_score in clf.grid_scores_:
            # print(grid_score[0])  # parameters
            # print(grid_score[1])  # mean
            # print(np.std(grid_score[2]))  # std
            print "%f, %f" % (grid_score[1], np.std(grid_score[2]))

        return clf.best_score_

    def gridSearchSingleSet(self):
        """
        Perform grid search using a single set
        Goals is to find the optimal classifier parameters
        :return:
        """

        datasetIdx = 1
        datasetId = self.datasetIds[datasetIdx]

        dataset = {
            'id': datasetId,
            'videoFile': os.path.join(self.projectDirectory, 'videos', datasetId+".MOV"),
            'cutouts': {
                'posDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'pos'),
                'negDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'neg')
            },
            'labelsFile': os.path.join(self.projectDirectory, 'labels', datasetId+"_output.txt"),
            'framesDir': os.path.join(self.projectDirectory, 'frames', datasetId),
        }

        foldsCount = 5
        objectIndices = self.getObjectsIndices(dataset['cutouts']['posDir'])
        foldsIndices = self.getFoldsIndices(objectIndices, foldsCount)

        negCutoutFiles = glob.glob(dataset['cutouts']['negDir']+"/*")
        negSamples = []
        for fileName in negCutoutFiles:
            negSamples.append({
                'fileName': fileName,
                'label': False,
                'features': self.getImageFileFeatures(fileName)
            })

        negSamplesPerFold = len(negSamples)/foldsCount

        samples = []
        foldIdx = 0
        for objectIndices in foldsIndices:
            posCutoutFiles = []
            for objectIndex in objectIndices:
                posCutoutFiles += (glob.glob(dataset['cutouts']['posDir'] + "/cutout_" + str(objectIndex) + "_*.png"))

            foldPosSamples = []
            for fileName in posCutoutFiles:
                foldPosSamples.append({
                    'fileName': fileName,
                    'label': True,
                    'features': self.getImageFileFeatures(fileName),
                    'foldIdx': foldIdx
                })

            foldNegSamples = negSamples[negSamplesPerFold*foldIdx:negSamplesPerFold*foldIdx + negSamplesPerFold]
            for foldNegSample in foldNegSamples:
                foldNegSample['foldIdx'] = foldIdx

            samples.extend(foldPosSamples+foldNegSamples)
            foldIdx += 1

        X = [sample['features'] for sample in samples]
        y = [sample['label'] for sample in samples]
        test_fold = [sample['foldIdx'] for sample in samples]
        ps = PredefinedSplit(test_fold=test_fold)

        # est = svm.SVC()
        # parameters = {'kernel':['rbf'], 'C':[3], 'gamma': [0.0625]}
        # parameters = {'kernel':['rbf'], 'C':[3]}

        # est = svm.SVC()
        # parameters = {'C': (1, 2, 3)}

        est = KNeighborsClassifier()
        parameters = {'n_neighbors': [1]}

        clf = GridSearchCV(
            estimator=est,
            param_grid=parameters,
            n_jobs=1,
            pre_dispatch='2*n_jobs',
            iid=False,
            refit=True,
            cv=ps
        )
        clf.fit(X, y)
        # pprint(clf.best_score_)
        # pprint(clf.best_params_)
        # print("Results:")
        # pprint(clf.grid_scores_)
        return clf.best_score_



    def getFoldsIndices(self, objectsIndices, foldsCount):
        """
        Return a list of folds, where each fold is a list of unique indices of objects
        Each object (e.g. a Cow) has a unique index

        :param objectsIndices:
        :param foldsCount:
        :return:
        """

        foldObjectsCount = [0]*foldsCount
        objectsCount = len(objectsIndices)
        while objectsCount > 0:
            for idx in range(foldsCount):
                if objectsCount > 0:
                    foldObjectsCount[idx] += 1
                    objectsCount -= 1
                else:
                    break

        foldsIndices = []
        lastIndex = -1
        for count in foldObjectsCount:
            foldIndex = []
            for idx in range(count):
                lastIndex += 1
                foldIndex.append(objectsIndices[lastIndex])

            foldsIndices.append(foldIndex)

        return foldsIndices

    def getTrainSamples(self, trainsetIdxs):
        allSamples = []

        for setIdx in trainsetIdxs:
            datasetId = self.datasetIds[setIdx]

            dataset = {
                'id': datasetId,
                'videoFile': os.path.join(self.projectDirectory, 'videos', datasetId+".MOV"),
                'cutouts': {
                    'posDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'pos'),
                    'negDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'neg')
                },
                'labelsFile': os.path.join(self.projectDirectory, 'labels', datasetId+"_output.txt"),
                'framesDir': os.path.join(self.projectDirectory, 'frames', datasetId),
            }

            posCutoutFiles = (glob.glob(dataset['cutouts']['posDir'] + "/*.png"))
            posLabels = [True] * len(posCutoutFiles)
            negCutoutFiles = (glob.glob(dataset['cutouts']['negDir'] + "/*.png"))
            negLabels = [False] * len(negCutoutFiles)
            samples = []
            for cutoutFile, label in zip(posCutoutFiles + negCutoutFiles, posLabels + negLabels):
                samples.append({
                    'fileName': cutoutFile,
                    'label': label,
                    'features': self.getImageFeatures(cv2.imread(cutoutFile)),
                    'foldIdx': setIdx
                })

            allSamples.extend(samples)

        return allSamples

    def trainFromSamples(self, samples):
        X = [sample['features'] for sample in samples]
        y = [sample['label'] for sample in samples]

        if self.featureType ==  "HOG":
            k=1
            C=3
            gamma=1000
        elif self.featureType == "HOG_HSV-HIST" or self.featureType == "HSV-HIST":
            k=3
            C=1
            gamma=100

        if self.classifierType == 'KNN':
            clsf = KNeighborsClassifier(n_neighbors=k)
        elif self.classifierType == 'SVM-RBF':
            clsf = svm.SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
        elif self.classifierType == 'SVM-LIN':
            clsf = svm.SVC(C=C, kernel='linear', probability=True)
        else:
            raise Exception("Uknown classifier type %s" % self.classifierType)



        clsf.fit(X, y)

        return clsf

    def test_detect(self, harvest=False):
        self.log.info("Performing detection test")

        trainsetIdxs = [0, 1, 2]
        testsetIdx = 1
        framesStep = 10  # take every frameSteps-th frame


        samples = self.getTrainSamples(trainsetIdxs)

        clsf = self.trainFromSamples(samples)  # train a classifier based on the given dataset samples
        datasetId = self.datasetIds[testsetIdx]  # get the dataset id given the dataset index
        videoFileName = os.path.join(self.projectDirectory, 'videos', '%s.MOV' % datasetId)
        annotationsFileName = os.path.join(self.projectDirectory, 'labels', "%s_output.txt" % datasetId)
        video = Video(videoFileName, annotationsFileName)
        video.cutoutSize = (100, 100)
        detectionsTruth = video.getAnnotations()  # get the labeled ground truth for this video

        self.log.debug("Using video %s" % videoFileName)

        self.suppressor = Suppressor()

        displayOptions = {
            'enabled': True,  # display anything at all
            'frame': True,  # video frame image
            'detections': False,  # detected objects
            'suppressed_detections': True,
            'sliding_window': False,
            'ground_truth': False
        }

        cap = cv2.VideoCapture(videoFileName)  # load the video file

        results = []  # results for each frame
        frameIdx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            if ((frameIdx+1) % framesStep) == 0:  # use every <frameStep>th frame
                self.log.debug("Frame %d" % frameIdx)

                truth = []
                for truthDetection in detectionsTruth[frameIdx]:
                    if truthDetection['lost'] is True:  # if the detection is not visible in this frame
                        continue

                    truthDetectionCutoutDimensions = video.getCutoutDimensions(truthDetection, frame.shape[:2])  # dimension of the truth in the frame
                    truth.append(truthDetectionCutoutDimensions)

                detections = self.detectObjects(frame, clsf, truth, display=displayOptions)
                result = self.matchDetections(detections, truth, frame.shape[:2])
                results.append(result)

                if harvest:
                    detectionsOverlapRatios = result['detectionsOverlapRatios']
                    newSamples = self.annotateDetections(frame, detections, detectionsOverlapRatios)
                    if len(newSamples):
                        samples.extend(newSamples)
                        clsf = self.trainFromSamples(samples)  # retrain a classifier based on the given dataset samples and new samples

            frameIdx += 1

        cap.release()

        totalScore = sum([result['score'] for result in results])
        totalPrecision = sum([result['precision'] for result in results])
        totalRecall = sum([result['recall'] for result in results])

        pprint(totalScore/len(results))
        pprint(totalPrecision/len(results))
        pprint(totalRecall/len(results))

    def suppressDetections(self, detections):
        """
        Suppress the detections using the suppressor

        :param detections:
        :return:
        """

        self.log.debug("Suppressing detections...")
        boxes = []
        for detection in detections:
            boxes.append(self.frameToCoordinates(detection['frame']))

        overlapThresh = 0.3  # how much the bounding boxes may overlap
        newBoxes = self.suppressor.non_max_suppression_fast(np.array(boxes), overlapThresh=overlapThresh)
        return [self.coordinatesToFrame(newBox) for newBox in newBoxes]

    def matchDetectionsOverlap(self, detections, truthDetections, frameSize):
        self.log.info("Calculating overlap for each detection..")

        truthCanvas = np.zeros((frameSize[0], frameSize[1], 1), np.uint8)
        for truthDetection in truthDetections:
            truthCanvas[truthDetection['ymin']:truthDetection['ymax'], truthDetection['xmin']:truthDetection['xmax']] = 255

        overlapRatios = []
        for detection in detections:
            detectionCanvas = np.zeros((frameSize[0], frameSize[1], 1), np.uint8)
            detectionCanvas[detection['ymin']:detection['ymax'], detection['xmin']:detection['xmax']] = 255

            canvas = cv2.bitwise_and(detectionCanvas, truthCanvas)
            detectionCount = np.count_nonzero(detectionCanvas)
            overlapCount = np.count_nonzero(canvas)
            overlap = (overlapCount*1.0)/detectionCount

            overlapRatios.append(overlap)

        return overlapRatios

    def matchDetections(self, detections, truthDetections, frameSize):
        """
        Match the found detections with the ground truth

        :param detections:
        :param truthDetections:
        :param frameSize:
        :return:
        """

        self.log.info("Matching detection with ground truth...")
        detectionCanvas = np.zeros((frameSize[0], frameSize[1], 1), np.uint8)
        for detection in detections:
            detectionCanvas[detection['ymin']:detection['ymax'], detection['xmin']:detection['xmax']] = 255

        truthCanvas = np.zeros((frameSize[0], frameSize[1], 1), np.uint8)
        for truthDetection in truthDetections:
            truthCanvas[truthDetection['ymin']:truthDetection['ymax'], truthDetection['xmin']:truthDetection['xmax']] = 255


        canvas = cv2.bitwise_and(detectionCanvas, truthCanvas)

        detectionCount = np.count_nonzero(detectionCanvas)
        truthCount = np.count_nonzero(truthCanvas)
        overlapCount = np.count_nonzero(canvas)

        self.log.debug("dc: %d, tc: %d, oc: %d" % (detectionCount, truthCount, overlapCount))

        if detectionCount + truthCount == 0:  # nothing detected and nothing to detect, so perfect
            score = 1
        elif detectionCount * truthCount == 0:  # detect nothing but should, or detect but should not
            score = 0
        else:
            score = overlapCount / math.sqrt(detectionCount*truthCount)
        self.log.info("Score: %f" % score)

        # calculate precision.
        # If there should be detected something, but nothing is detected, then, precision is always 0%
        # If nothing should be detected and there is nothing detected, then precision is always 100%
        if detectionCount == 0 and truthCount != 0:
            precision = 0
        elif detectionCount == 0:
            precision = 1
        else:
            precision = float(overlapCount) / detectionCount

        # calculate recall. If there is nothing to detect, then recall is always 100%
        if truthCount == 0:
            recall = 1
        else:
            recall = float(overlapCount) / truthCount

        return {
            'score': score,
            'precision': precision,
            'recall': recall,
            'detectionsOverlapRatios': self.matchDetectionsOverlap(detections, truthDetections, frameSize)
        }

    def frameToCoordinates(self, frame):
        return (frame['xmin'], frame['ymin'], frame['xmax'], frame['ymax'])

    def coordinatesToFrame(self, coordinates):
        return {
            'xmin': coordinates[0],
            'xmax': coordinates[2],
            'ymin': coordinates[1],
            'ymax': coordinates[3]
        }

    def detectObjects(self, img, classifier, truthDetections, display=None):
        self.log.debug("Start detecting objects in image...")

        if display and not display['enabled']:
            display = None

        imgHeight, imgWidth = img.shape[:2]
        windowSize = (100, 100)
        windowY = 0
        windowX = 0
        windowStep = 25

        displayImg = img.copy()

        foundDetections = []

        # find detections using sliding window
        while (windowY + windowSize[0]) < imgHeight:
            while (windowX + windowSize[1]) < imgWidth:
                frame = {
                    'xmin': windowX,
                    'xmax': windowX+windowSize[0],
                    'ymin': windowY,
                    'ymax': windowY+windowSize[1]
                }

                window = img[windowY:windowY+windowSize[0], windowX:windowX+windowSize[1]]
                features = self.getImageFeatures(window)
                result = classifier.predict([features])
                objectFound = result[0]

                if objectFound:
                    foundDetections.append({
                        'frame': frame,
                        'image': window,
                        'features': features
                    })

                    # display the detected objects
                    if display and display['detections']:
                        cv2.rectangle(displayImg, (windowX, windowY), (windowX+windowSize[0], windowY+windowSize[1]), (255, 0, 0), 3)
                else:
                    # display the sliding window
                    if display and display['sliding_window']:
                        cv2.rectangle(displayImg, (windowX, windowY), (windowX+windowSize[0], windowY+windowSize[1]), (75, 75, 75), 1)
                        # cv2.rectangle(displayImg, (windowX, windowY), (windowX+windowSize[0], windowY+windowSize[1]), (255, 255, 255), 2)
                        # imageFileName = "slidingwindow_ss/%s.png" % str(time.time())
                        # cv2.imwrite(imageFileName, displayImg)


                windowX += windowStep

            windowX = 0
            windowY += windowStep

        self.log.debug("Found %d detections" % len(foundDetections))

        # display labeled truth detections
        for frameDetection in truthDetections:
            if display and display['ground_truth']:
                cv2.rectangle(displayImg, (frameDetection['xmin'], frameDetection['ymin']), (frameDetection['xmax'], frameDetection['ymax']), (0, 255, 0), 1)
                # cv2.rectangle(displayImg, (frameDetection['xmin'], frameDetection['ymin']), (frameDetection['xmax'], frameDetection['ymax']), (255, 0, 0), 3)

            # if self.RESIZE_CUTOUTS:
            #     cutout = cv2.resize(cutout, self.cutoutSize)

        # get the detections using the non-max suppression algorithm
        suppressedDetections = self.suppressDetections(foundDetections)
        self.log.debug("Found %d suppressed detections" % len(suppressedDetections))

        if display and display['suppressed_detections']:
            for suppressedDetection in suppressedDetections:
                cv2.rectangle(displayImg, (suppressedDetection['xmin'], suppressedDetection['ymin']), (suppressedDetection['xmax'], suppressedDetection['ymax']), (255, 255, 255), 2)
                suppressedCutout = img[suppressedDetection['ymin']:suppressedDetection['ymax'], suppressedDetection['xmin']:suppressedDetection['xmax']]
                imageFileName = "supressed_cutouts/%s.png" % str(time.time())
                cv2.imwrite(imageFileName, suppressedCutout)


        if display:
            resizedImg = cv2.resize(displayImg, (int(displayImg.shape[1]/1.5), int(displayImg.shape[0]/1.5)))
            cv2.imshow('Video', resizedImg)
            # imageFileName = "screenshots_truth/%s.png" % str(time.time())
            # cv2.imwrite(imageFileName, displayImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()

        return suppressedDetections

    def annotateDetections(self, frame, detectionsCutoutDimensions, detectionsOverlapRatios, autoAnnotate=True):

        overlapThreshold = 0.3

        # shuffle(detectionsCutoutDimensions)
        detections = []
        for idx, cutoutDimensions in enumerate(detectionsCutoutDimensions):
            cutout = frame[cutoutDimensions['ymin']:cutoutDimensions['ymax'], cutoutDimensions['xmin']:cutoutDimensions['xmax']]

            if autoAnnotate:
                label = detectionsOverlapRatios[idx] >= overlapThreshold
            else:
                cv2.imshow("Detection", cutout)
                label = None
                while label is None:
                    key = cv2.waitKey(0)
                    if key == ord('p'):
                        label = True
                    elif key == ord('n'):
                        label = False
                    else:
                        print("Press 'p' if this is a cow. Else press 'n'")

            detections.append({
                'image': cutout,
                'features': self.getImageFeatures(cutout),
                'label': label
            })

        return detections

    def getSampleFeatures(self, sample):
        image = cv2.imread(sample['fileName'])

        return self.getImageFeatures(image)

    def getImageFileFeatures(self, imageFileName):
        image = cv2.imread(imageFileName)

        return self.getImageFeatures(image)

    def getImageFeatures(self, image):
        type = self.featureType
        if not type in self.featureOptions:
            raise ValueError("Feature options for type %s not specified" % type)

        featureOptions = self.featureOptions[type]

        if type == 'BGR-HIST':
            hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            return hist.flatten()
        elif type == 'HSV-HIST':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv],
                featureOptions['channels'],
                featureOptions['mask'],
                featureOptions['histSize'],
                featureOptions['ranges']
            )

            return hist.flatten()/(hsv.shape[0]*hsv.shape[1])

        elif type == 'HOG':
            imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fd = hog(
                imageGray,
                featureOptions['orientations'],
                featureOptions['pixels_per_cell'],
                featureOptions['cells_per_block'],
            )
            return fd/featureOptions["orientations"]
        elif type == "HOG_HSV-HIST":
            imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fd = hog(
                imageGray,
                featureOptions['orientations'],
                featureOptions['pixels_per_cell'],
                featureOptions['cells_per_block'],
            )

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv],
                featureOptions['channels'],
                featureOptions['mask'],
                featureOptions['histSize'],
                featureOptions['ranges']
            )

            histFlattened = hist.flatten()

            combo = list(fd/featureOptions["orientations"]) + list(histFlattened/(hsv.shape[0]*hsv.shape[1]))
            return combo
        else:
            raise Exception("Unsupported type %s" % type)

    def getTrainKeys(self, probs, limit=None):
        """
        Get the dict keys (indices) of the samples that need to be trained to improve the classifier most

        :param probs: list of samples with probabilities for each class, provided by the classifier
        :param limit: how many to return
        :return: the indices of the samples with lowest standard deviation between class probabilities
        """

        stds = np.std(probs, axis=1)
        sortedKeys = np.argsort(stds)

        if limit:
            return sortedKeys[:limit]

        return sortedKeys


    def activeLearningTest(self, active=True):
        """
        Perform active learning test using a single set
        :return:
        """

        print("Building dataset")
        datasetIdx = 1
        datasetId = self.datasetIds[datasetIdx]

        dataset = {
            'id': datasetId,
            'videoFile': os.path.join(self.projectDirectory, 'videos', datasetId+".MOV"),
            'cutouts': {
                'posDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'pos'),
                'negDir': os.path.join(self.projectDirectory, 'cutouts', datasetId, 'neg')
            },
            'labelsFile': os.path.join(self.projectDirectory, 'labels', datasetId+"_output.txt"),
            'framesDir': os.path.join(self.projectDirectory, 'frames', datasetId),
        }

        trainFactor = 0.9  # factor of the dataset that is used for training, opposite to testing
        objectIndices = self.getObjectsIndices(dataset['cutouts']['posDir'])  # indices of the unique objects in the dataset (different cows e.g.)
        trainsetSize = int(len(objectIndices)*trainFactor)  # amount of unique objects that are used for training, based on the trainFactor
        trainIndices = objectIndices[:trainsetSize]  # actual objects indices that are used for training
        testIndices = objectIndices[trainsetSize:]  # actual object indices that are used for testing

        # cutout image that are used
        posTrainCutoutFiles = []
        for objectIndex in trainIndices:
            posTrainCutoutFiles += (glob.glob(dataset['cutouts']['posDir'] + "/cutout_" + str(objectIndex) + "_*.png"))
        posTestCutoutFiles = []
        for objectIndex in testIndices:
            posTestCutoutFiles += (glob.glob(dataset['cutouts']['posDir'] + "/cutout_" + str(objectIndex) + "_*.png"))

        trainSamples = []
        for fileName in posTrainCutoutFiles:
            trainSamples.append({
                'fileName': fileName,
                'label': True,
                'features': self.getImageFileFeatures(fileName),
            })

        testSamples = []
        for fileName in posTestCutoutFiles:
            trainSamples.append({
                'fileName': fileName,
                'label': True,
                'features': self.getImageFileFeatures(fileName),
            })

        negCutoutFiles = glob.glob(dataset['cutouts']['negDir']+"/*")
        negSamples = []
        for fileName in negCutoutFiles:
            negSamples.append({
                'fileName': fileName,
                'label': False,
                'features': self.getImageFileFeatures(fileName)
            })

        trainSamples.extend(negSamples[:trainsetSize])
        testSamples.extend(negSamples[trainsetSize:])

        shuffle(trainSamples)
        shuffle(testSamples)


        trainStep = int(0.1*len(trainSamples))
        initTrainSamples = trainSamples[:trainStep]
        remainingTrainSamples = trainSamples[trainStep:]


        scores = []
        while len(remainingTrainSamples) > 0:
            clf = self.trainFromSamples(initTrainSamples)

            if active:
                probabilities = clf.predict_proba([sample['features'] for sample in remainingTrainSamples])

                # get the samples used for the next training round
                trainKeys = self.getTrainKeys(probabilities, limit=trainStep)
            else:
                trainKeys = min(range(trainStep), range(len(remainingTrainSamples)))

            for trainKey in trainKeys:
                initTrainSamples.append(remainingTrainSamples[trainKey])

            remainingTrainSamples = np.delete(remainingTrainSamples, trainKeys, 0)

            score = clf.score([sample['features'] for sample in testSamples],
                              [sample['label'] for sample in testSamples])
            scores.append(score)

        return scores

    def activeLearningTestExp(self, active=True):
        results = None
        runs = 20
        for run in range(runs):
            result = detector.activeLearningTest(active)
            if not results:
                results = result
            else:
                results = [x + y for x, y in zip(results, result)]

        pprint([x/runs for x in results])


if __name__ == "__main__":
    detector = Detector()
    # detector.parameterSweep()
    # detector.gridSearchSingleSet()
    # detector.gridSearch()
    detector.test_detect(harvest=False)
    # detector.activeLearningTestExp(True)
    # detector.activeLearningTestExp(False)



