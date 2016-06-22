# UAV-ODF
Unmanned Aerial Vehicle Object Detection Framework

## Installation requirements


## Dataset
The complete used dataset can be found at https://rixmit.stackstorage.com/index.php/s/rb1eTkzFQk0gr1y
This dataset contains the following data:
1. videos
   The recorded videos
2. frames
   Frames that are extracted from the videos for Vatic
3. labels
   The annotations for the videos that are made using Vatic
4. cutouts
   The extracted cutouts for each of the videos, made using the labels


## Vatic
Vatic is used to annotate the recorded videos.
Installation and documenations on Vatic can be found at https://github.com/cvondrick/vatic
the 'dataset_scripts' directory contains some basic scripts to:
1. crop a video ('cut.sh')
2. input a video in Vatic ('vatic_import.sh')
3. export annotations from Vatic ('vatic_export.sh')