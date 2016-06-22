import sys
import os
import time
import csv
import random
from pprint import pprint

import numpy as np
import cv2

from Run import Run

__author__ = 'rik'

class Video(Run):

    detections = None
    cutoutRootDir = None
    cutoutPosDir = None
    cutoutNegDir = None
    cutoutSize = (50, 50)
    negExamplesPerFrame = 50
    framesStep = 25  # use every <frameStep>th frame

    RESIZE_CUTOUTS = False

    def __init__(self, videoFileName, annotationsFileName):
        if not os.path.isfile(videoFileName):
            raise Exception("No such file %s" % videoFileName)
        if not os.path.isfile(annotationsFileName):
            raise Exception("No such file %s" % annotationsFileName)

        self.videoFileName = videoFileName
        self.videoFileNameBase = os.path.basename(os.path.normpath(self.videoFileName))
        self.annotationsFileName = annotationsFileName

    def load(self):
        print ("Loading video ", self.videoFileName, self.annotationsFileName)
        self.detections = self.getAnnotations()

    def getAnnotations(self):
        reader = csv.reader(open(self.annotationsFileName, 'rb'), delimiter=' ')

        detections = {}
        for row in reader:
            if int(row[5]) not in detections:
                detections[int(row[5])] = []

            detections[int(row[5])].append({
                      'xmin': int(round(float(row[1]))),
                      'ymin': int(round(float(row[2]))),
                      'xmax': int(round(float(row[3]))),
                      'ymax': int(round(float(row[4]))),
                      'value': int(row[0]),
                      'lost': bool(int(row[6]))
        })

        return detections

    def extractObjects(self, cutoutDir, extractPositives=True, extractNegatives=True, convertToGray=False):
        if (not extractPositives) and (not extractNegatives):
            print("No task given...")
            return

        # make sue the root dir exists
        self.cutoutRootDir = cutoutDir
        if not os.path.isdir(self.cutoutRootDir):
            os.mkdir(self.cutoutRootDir)

        if extractPositives:
            self.cutoutPosDir = os.path.join(self.cutoutRootDir, "pos")
            if not os.path.isdir(self.cutoutPosDir):
                os.mkdir(self.cutoutPosDir)

        if extractNegatives:
            self.cutoutNegDir = os.path.join(self.cutoutRootDir, "neg")
            if not os.path.isdir(self.cutoutNegDir):
                os.mkdir(self.cutoutNegDir)

        cap = cv2.VideoCapture(self.videoFileName)

        frameIdx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            if convertToGray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ((frameIdx+1) % self.framesStep) == 0:
                print("Start cutting out for frame %d" % frameIdx)
                if extractPositives:
                    self.cutoutPositiveDetections(frame, frameIdx)
                if extractNegatives:
                    self.cutoutNegativeDetections(frame, frameIdx)

            frameIdx += 1

        cap.release()

    def getCutoutDimensions(self, frameDetection, (h, w)):
        """
        Get the cutout coordinates from a given Vatic frameDetection.
        h, w denote the size of the video frame

        :param frameDetection:
        :return:
        """

        yCenter = int((frameDetection['ymax'] + frameDetection['ymin']) / 2)
        xCenter = int((frameDetection['xmax'] + frameDetection['xmin']) / 2)

        halfY = int(self.cutoutSize[0]/2)
        halfX = int(self.cutoutSize[1]/2)

        # adjust for edge cases (at the edge case the cutout will be still of cutout size)
        yCenter = max(halfY, yCenter)
        xCenter = max(halfX, xCenter)

        yCenter = min(h-halfY, yCenter)
        xCenter = min(w-halfX, xCenter)

        newDimensions = {
            'ymin': yCenter - halfY,
            'ymax': yCenter + halfY,
            'xmin': xCenter - halfX,
            'xmax': xCenter + halfX,
            'value': frameDetection['value'],
            'lost': frameDetection['lost']
        }

        assert(newDimensions['ymax'] - newDimensions['ymin'] == self.cutoutSize[0])
        assert(newDimensions['xmax'] - newDimensions['xmin'] == self.cutoutSize[1])

        return newDimensions


    def cutoutPositiveDetections(self, img, frameIdx):
        # get detections for this frame
        if frameIdx in self.detections:
            frameDetections = self.detections[frameIdx]

            for frameDetection in frameDetections:
                if frameDetection['lost'] is True:  # if the detection is not visible in this frame
                    continue

                cutoutDimensions = self.getCutoutDimensions(frameDetection, img.shape[:2])

                cutout = img[cutoutDimensions['ymin']:cutoutDimensions['ymax'], cutoutDimensions['xmin']:cutoutDimensions['xmax']]
                cutoutFileName = os.path.join(self.cutoutPosDir, "cutout_%d_%d.png" % (cutoutDimensions['value'], frameIdx))

                if self.RESIZE_CUTOUTS:
                    cutout = cv2.resize(cutout, self.cutoutSize)

                print("Writing pos detection %s" % cutoutFileName)
                cv2.imwrite(cutoutFileName, cutout)


    def cutoutNegativeDetections(self, img, frameIdx):
        demo = False # shows where the negatives are cutout (if True)
        if demo:
            demoImg = img.copy()

        negImg = img.copy()

        mask = np.zeros(img.shape[:2], np.uint8)

        # get detections for this frame
        if frameIdx in self.detections:
            frameDetections = self.detections[frameIdx]

            for frameDetection in frameDetections:
                if frameDetection['lost'] is True:  # if the detection is not visible in this frame
                    continue

                # blackout positive detection
                negImg[frameDetection['ymin']:frameDetection['ymax'], frameDetection['xmin']:frameDetection['xmax']] = 0

                mask[frameDetection['ymin']:frameDetection['ymax'], frameDetection['xmin']:frameDetection['xmax']] = True

                if demo:
                    cv2.rectangle(demoImg, (frameDetection['xmin'], frameDetection['ymin']), (frameDetection['xmax'], frameDetection['ymax']), (255, 255, 255), 2)


        # get negative examples
        exampleCount = 0
        falseNegs = 0 # amount of times we didn't find a good neg cutout in a row.
        while exampleCount < self.negExamplesPerFrame:
            xMin = random.randint(0, negImg.shape[1]-self.cutoutSize[1])
            yMin = random.randint(0, negImg.shape[0]-self.cutoutSize[0])
            negCutout = negImg[yMin:yMin+self.cutoutSize[0], xMin:xMin+self.cutoutSize[1]]
            if self.goodNegCutout(mask[yMin:yMin+self.cutoutSize[0], xMin:xMin+self.cutoutSize[1]]):

                if demo:
                    cv2.rectangle(demoImg, (xMin, yMin), (xMin+self.cutoutSize[1], yMin+self.cutoutSize[0]), (255, 0, 0), 2)

                # find a correct name (use exampleidx that has not been used yet)
                negCutoutFileName = None
                exampleIdx = 0
                while negCutoutFileName is None or os.path.isfile(negCutoutFileName):
                    negCutoutFileName = os.path.join(self.cutoutNegDir, "cutout_%d_%d.png" % (exampleIdx, frameIdx))
                    exampleIdx += 1

                print("Writing neg detection %s" % negCutoutFileName)
                cv2.imwrite(negCutoutFileName, negCutout)

                exampleCount += 1
                falseNegs == 0
            else:
                falseNegs += 1
                if falseNegs == 10: # Couldn't find a good neg...
                    break

        masked = cv2.bitwise_and(img, img, mask=mask)

        if demo:
            resizedImg = cv2.resize(demoImg, (int(demoImg.shape[1]/1.5), int(demoImg.shape[0]/1.5)))
            cv2.imshow('Video', resizedImg)
            imageFileName = "%s.png" % str(time.time())
            cv2.imwrite(imageFileName, resizedImg)
        else:
            cv2.imshow('Video', negImg)
            cv2.imshow('Video', masked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()

    def goodNegCutout(self, cutoutMask):
        return not np.count_nonzero(cutoutMask)

    def show(self, includeAnnotations=True):
        cap = cv2.VideoCapture(self.videoFileName)

        frameIdx = 0
        while cap.isOpened():
            ret, img = cap.read()

            # get detections for this frame
            if includeAnnotations and (frameIdx in self.detections):
                frameDetections = self.detections[frameIdx]

                for frameDetection in frameDetections:
                    if frameDetection['lost'] is True:  # if the detection is not visible in this frame
                        continue

                    colour = (255, 0, 0)
                    cv2.rectangle(img, (frameDetection['xmin'], frameDetection['ymin']), (frameDetection['xmax'], frameDetection['ymax']), colour, 2)
                    cv2.putText(img, "Value " + str(frameDetection['value']), (frameDetection['xmin'], frameDetection['ymax'] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow('Video', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frameIdx += 1

        cap.release()
        cv2.destroyAllWindows()
