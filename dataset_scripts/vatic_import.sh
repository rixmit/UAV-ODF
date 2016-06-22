#!/bin/sh

VATIC_DIR="$1"
PROJECT_DIR="$2"
VIDEO_IDENTIFIER="$3"

OBJECTS="Cow"
TITLE="Cow dataset"
DESCRIPTION="Cows in Drenthe, Netherlands"

echo "Using Vatic directory $VATIC_DIR"
cd $VATIC_DIR
VIDEO_FILE="$PROJECT_DIR/videos/$VIDEO_IDENTIFIER.MOV"
FRAMES_DIR="$PROJECT_DIR/frames/$VIDEO_IDENTIFIER"
echo "Using video $VIDEO_FILE"
echo "Using frames dir $FRAMES_DIR"
echo "Building frames from video..."
turkic extract $VIDEO_FILE $FRAMES_DIR --no-resize
echo "Done"
echo "Creating Vatic set from frames"
turkic load $VIDEO_IDENTIFIER $FRAMES_DIR $OBJECTS --title "$TITLE" --description "$DESCRIPTION" --offline

turkic publish --offline
