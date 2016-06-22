#!/bin/sh

# $0 is the script name
# $1 is the input filename
# $2 is the output filename
# $3 is the start time ARG
# $4 is the end time

INPUTFILENAME="$1"
TMPFILENAME="tmp_file.MOV"
OUTPUTFILENAME="$2"
DURATION=`expr $4 - $3`

ffmpeg -i $INPUTFILENAME -ss $3 -c copy $TMPFILENAME
ffmpeg -i $TMPFILENAME -t $DURATION -c copy $OUTPUTFILENAME
rm $TMPFILENAME
