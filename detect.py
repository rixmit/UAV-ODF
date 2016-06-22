import sys
import os
import glob
from pprint import pprint
import re
from random import shuffle

from sklearn.neighbors import KNeighborsClassifier
import cv2

__author__ = 'rik'

def annotateDetections(detections):
    # shuffle(detections)
    for detection in detections:
        cv2.imshow("Detection", detection['image'])
        label = None
        while label is None:
            key = cv2.waitKey(0)
            if key == ord('p'):
                label = True
            elif key == ord('n'):
                label = False
            else:
                print("Press 'p' if this is a cow. Else press 'n'")

        detection['label'] = label

    return detections

def detectObjects(img, classifier, annotate=False):
    print("Start detecting objects in image...")

    imgHeight, imgWidth = img.shape[:2]
    windowSize = (100, 100)
    windowY = 0
    windowX = 0
    windowStep = 50

    displayImg = img.copy()

    foundDetections = []

    while (windowY + windowSize[0]) < imgHeight:
        while (windowX + windowSize[1]) < imgWidth:
            window = img[windowY:windowY+windowSize[0], windowX:windowX+windowSize[1]]
            hist = cv2.calcHist([window], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            features = hist.flatten()
            result = classifier.predict([features])
            objectFound = result[0]

            if objectFound:
                foundDetections.append({
                    'image': window,
                    'features': features
                })
                cv2.rectangle(displayImg, (windowX, windowY), (windowX+windowSize[0], windowY+windowSize[1]), (255, 0, 0), 2)
            else:
                cv2.rectangle(displayImg, (windowX, windowY), (windowX+windowSize[0], windowY+windowSize[1]), (75, 75, 75), 1)


            windowX += windowStep

        windowX = 0
        windowY += windowStep

    cv2.imshow('Video', displayImg)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()

    if annotate:
        return annotateDetections(foundDetections)
    else:
        return []

def getSampleFeatures(sample):
    image = cv2.imread(sample['fileName'])
    hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])

    return hist.flatten()

def getSamplesNaiveRandom():
    cutoutsDirectory = "/home/rik/DroneprojectData/dataset/cows/cutouts"
    posPath = os.path.join(cutoutsDirectory, "pos")
    negPath = os.path.join(cutoutsDirectory, "neg")


    posCutoutFiles = map(lambda x: os.path.join(posPath, x), os.listdir(posPath))
    negCutoutFiles = map(lambda x: os.path.join(negPath, x), os.listdir(negPath))

    samples = []
    for fileName in posCutoutFiles:
        samples.append({
            'fileName': fileName,
            'label': True
        })
    for fileName in negCutoutFiles:
        samples.append({
            'fileName': fileName,
            'label': False
        })

    for sample in samples:
        print(sample['fileName'])
        sample['features'] = getSampleFeatures(sample)

    # randomize the list of samples
    shuffle(samples)

    # determine how many samples to use for training and testing
    samplesCount = len(samples)
    trainFactor = 0.8
    trainCount = int(round((samplesCount*trainFactor)))

    trainSamples = samples[:trainCount]
    testSamples = samples[trainCount:]

    return trainSamples, testSamples

def getSamplesSmartRandom():
    cutoutsDirectory = "/home/rik/DroneprojectData/dataset/cows/cutouts"
    posPath = os.path.join(cutoutsDirectory, "pos")
    negPath = os.path.join(cutoutsDirectory, "neg")


    posTrainCutoutFiles = []
    posTestCutoutFiles = []
    for cowIdx in range(70):
        posTrainCutoutFiles += (glob.glob(posPath + "/cutout_" + str(cowIdx) + "_*.png"))

    for cowIdx in range(70, 86):
        posTestCutoutFiles += (glob.glob(posPath + "/cutout_" + str(cowIdx) + "_*.png"))

    trainSamples = []
    testSamples = []
    for fileName in posTrainCutoutFiles:
        trainSamples.append({
            'fileName': fileName,
            'label': True
        })
    for fileName in posTestCutoutFiles:
        testSamples.append({
            'fileName': fileName,
            'label': True
        })

    negCutoutFiles = glob.glob(negPath+"/*")
    negSamples = []
    for fileName in negCutoutFiles:
        negSamples.append({
            'fileName': fileName,
            'label': False
        })

    # determine how many samples to use for training and testing
    samplesCount = len(negSamples)
    trainFactor = 0.8
    trainCount = int(round((samplesCount*trainFactor)))

    trainSamples += negSamples[:trainCount]
    testSamples += negSamples[trainCount:]

    for sample in trainSamples:
        sample['features'] = getSampleFeatures(sample)
    for sample in testSamples:
        sample['features'] = getSampleFeatures(sample)


    shuffle(trainSamples)
    shuffle(testSamples)

    return trainSamples, testSamples


trainSamples, testSamples = getSamplesSmartRandom()


trainFeatures = []
trainLabels = []
for sample in trainSamples:
    trainFeatures.append(sample['features'])
    trainLabels.append(sample['label'])

testFeatures = []
testLabels = []
for sample in testSamples:
    testFeatures.append(sample['features'])
    testLabels.append(sample['label'])

nbrs = KNeighborsClassifier(n_neighbors=3)
nbrs.fit(trainFeatures, trainLabels)

# res = nbrs.score(testFeatures, testLabels)
# pprint(res)
# sys.exit()

videoFileName = "/home/rik/DroneprojectData/dataset/cows/DJI_0007_cut_22-65.MOV"
# videoFileName = "/home/rik/DroneprojectData/dataset/cows/DJI_0005_cut_233-244.MOV"
cap = cv2.VideoCapture(videoFileName)

frameIdx = 0
framesStep = 5
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break

    if ((frameIdx+1) % framesStep) == 0: # use every <frameStep>th frame
        correct = None
        # while correct is None or (correct < 0.5 and len(detections) > 0):
        activeLearning = True
        detections = detectObjects(frame, nbrs, annotate=activeLearning)
        correct = 0.0
        for detection in detections:
            trainFeatures.append(detection['features'])
            trainLabels.append(detection['label'])
            correct += float(detection['label']) / float(len(detections))
        nbrs.fit(trainFeatures, trainLabels)
        print(correct)
    frameIdx += 1

cap.release()


