import argparse

from Video import Video

__author__ = 'rik'


parser = argparse.ArgumentParser(description='Creating cutous from video using Vatic annotations.')
parser.add_argument('videofile', help='Location of the video file')
parser.add_argument('annotationsfile', help='Location of the annotations file')
parser.add_argument('outputdir', help='Location of the directory to store the cutouts')
parser.add_argument('--cutoutsize', help='Dimensions of the cutouts.', default=(100, 100))
parser.add_argument('--framestep', help='Use every <framestep>th frame.', default=25)
parser.add_argument('--negativescount', help='Amount of negatives to cutout for each frame.', default=50)
parser.add_argument('--pos', action='store_false', help="Whether to exclude extracting positives")
parser.add_argument('--neg', action='store_false', help="Whether to exclude extracting negatives")

args = parser.parse_args()

videoFileName = args.videofile
cutoutsDirectory = args.outputdir
annotationsFileName = args.annotationsfile

video = Video(videoFileName, annotationsFileName)

video.load()

video.cutoutSize = args.cutoutsize
video.negExamplesPerFrame = args.negativescount
video.framesStep = args.framestep


video.extractObjects(cutoutsDirectory, extractPositives=args.pos, extractNegatives=args.neg)
