#!/bin/sh

VATIC_DIR="$1"
PROJECT_DIR="$2"
VIDEO_IDENTIFIER="$3"


OUTPUT_FILE="$PROJECT_DIR/labels/${VIDEO_IDENTIFIER}_output.txt"

echo "Using Vatic directory $VATIC_DIR"
cd $VATIC_DIR
turkic dump $VIDEO_IDENTIFIER -o $OUTPUT_FILE
