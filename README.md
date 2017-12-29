# Marker based nonoverlapping calibration
This is the code for paper "Marker based simple non-overlapping camera calibration" on ICIP2016.
You can check the paper for the work flow of our method.

## Requeriments:

PCL 1.7 (pointclouds.org)

OpenCV 2.4.13 (opencv.org)

Aruco 1.3.0 
You can find it on "https://sourceforge.net/projects/aruco/files/" under the "oldversions".

Note that the leatest version may not work for our code, though we haven't tested yet. 

## Build:
    mkdir build
    cd build
    cmake ..
    make

## Usage:

"boardM2TbyList" for calculating transformation between marker and target camera.

"boardT2TbyList" for calculating transformation between target cameras.

"synctestboardbyList" and "synctestT2TbyList" are for synthetic test, you can ignore them if not interested in. They may not able to work on your environment.

Use "boardM2TbyList" to find the transformation between marker and camera first, than you can use "boardT2TbyList" to find the transformation between cameras.

## Preparation:

This section is for calibrating your own cameras, but before do this you may want to have a try with our data first (check the Example section).

1, Calibrate your cameras with OpenCV for intrinsic. We prefer to use OpenCV's "cpp-example-calibration" to do this.

Use "cpp-example-imagelist_creator" to generate a list of your images, then feed it as parameter.
For the chessboard pattern we prefer you to use a 9 by 6, 2.54cm (1 inch) unit chessboard. This is what we used and hard coded in the cpp file. Note that when the printer printing the chessboard (and marker), it may rescale your image. Make sure the size it correct.

2, Prepare marker and attach to target camera. Use Aruco to do this.

    /aruco-1.3.0/build/utils/aruco_create_board 3:3 board.png board.yml 200 1
This is our setting, you can change it but you also need to change the code since we hard coded those parameters.

3, Prepare images for non overlapping calibration, ensure that:

in target camera's view the chessboard is visibile;
in support camera's view both chessboard and marker are visibile.

4, Calibrate marker to target camera first, result is saved in marker's config file (board.yml). Then calibrate the target to target camera, result is saved in T2Tresult.yml. 

    
## Example

We added a simple dataset to test if the code works fine.

1, calibrate first marker to target first camera.

    cp ./sample_data/board0/board.yml ./build/
Move the config file of your marker.

    cd ./build
    ./boardM2TbyList ../sample_data/68.yml ../sample_data/iphone.yml ../sample_data/non_68/target.yaml ../sample_data/non_68/support.yaml
    cp board.yml board68.yml
Now your marker to target camera's transformation are in marker's config file.

2, Do the same to another marker-camera pair.

    cp ./sample_data/board3/board.yml ./build/
    cd ./build
    ./boardM2TbyList ../sample_data/83.yml ../sample_data/iphone.yml ../sample_data/non_83/target.yaml ../sample_data/non_83/support.yaml
    cp board.yml board83.yml

3, calibrate between cameras

    ./boardT2TbyList ../sample_data/iphone.yml ../sample_data/noncalibration/support.yaml board83.yml board68.yml

