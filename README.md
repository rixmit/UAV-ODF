# UAV-ODF
Unmanned Aerial Vehicle Object Detection Framework. Scripts and tools for detecting (natural) objects in aerial video streams.

## Dependancies
- OS packages:
 - Python 2.7
 - Pip
 - freetype-devel (Fedora) / libfreetype6-dev (Ubuntu)
 - gstreamer1-libav
 - OpenCV 2.4 or OpenCV 3 (both should work)
 - Python bindings for OpenCV

For Fedora and OpenCV 2.4, installing the *opencv-python* packages worked for me.
For Ubuntu and OpenCV 3, see http://rodrigoberriel.com/2014/10/installing-opencv-3-0-0-on-ubuntu-14-04/

- Pip packages:
 - scikit-image
 - scikit-learn
 - numpy-1.11.0
 - setuptools-23.0.0

Different package versions may work as well.

## Usage

### Prepare datasets
To use the framework on your own videos, you may first want to:

1. Import and annotate your recorded videos in Vatic (see Vatic section)
2. Extract cutouts from the Vatic annotations (use the *cutout.py* script)

The available dataset however (see section below) provides the required resources for demonstrating the framework on a couple of prerecorded videos.

### Tests and experiments
Several scripts are available for various exeriments

1. **detect.py** Start detecting objects in a video recording
2. **activelearning.py** Perform an active learning test
3. **featureparametersweep.py** Find the optimal feature descriptor parameters
4. **gridsearch.py** Find optimal classifier parameters


## Dataset
The complete used dataset can be found at https://rixmit.stackstorage.com/index.php/s/rb1eTkzFQk0gr1y
This dataset contains the following data:

1. **videos**
   The recorded videos
2. **frames**
   Frames that are extracted from the videos for Vatic
3. **labels**
   The annotations for the videos that are made using Vatic
4. **cutouts**
   The extracted cutouts for each of the videos, made using the labels


## Vatic
Vatic is used to annotate the recorded videos. The videos in the dataset are already annotated (see the *labels* directory in the dataset), but if you would like to add now videos to the dataset, then Vatic can be used for annotating these.
Installation and documenations on Vatic can be found at https://github.com/cvondrick/vatic
the *dataset_scripts* directory contains some basic scripts to:

1. crop a video (cut.sh)
2. input a video in Vatic (vatic_import.sh)
3. export annotations from Vatic (vatic_export.sh)