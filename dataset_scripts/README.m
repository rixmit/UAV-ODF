#Preparing dataset vor Vatic
These are some command that can be run to prepare a video for being labeled in Vatic. Most of this is explained in the Vatic documentation as well: https://github.com/cvondrick/vatic

(you may need to runt he commands from the Vatic installation directory)

Extract frames from the video:
`turkic extract <dataset_dir>/dataset/cows/videos/DJI_0007_cut_22-65.MOV <dataset_dir>/dataset/cows/frames/DJI_0007 --no-resize`

Create a set in Vatic from the frames with a unique identifier
`turkic load COW0007 <dataset_dir>/cows/frames/DJI_0007 Cow --title "Cow dataset" --description "Cows in Drenthe, Netherlands" --offline`

Publish the set
`turkic publish --offline`
  
Remove set
`turkic delete <identifier>``
  
Write the labels to file
`turkic dump COW0007 -o <output_dir>/vatic_output.txt`