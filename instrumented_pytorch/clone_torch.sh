#!/bin/bash

## clone the repo and all (submodules)
rm -r pytorch -f
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout 1a71eb647bd1502e301a94bc0009752f8a4278c7
git submodule sync
git submodule update --init --recursive
# install deps into the environment
pip install -r requirements.txt
cd ..

####### 3. apply patch to collxect instrumentation!!!
PATCH_FILE_1="cmake_patch1.patch" # Define the location of the patch file for the CMakeLists.txt
PATCH_FILE_2="cmake_patch2.patch" # Define the location of the patch file for the caffe2/CMakeLists.txt
TARGET_DIR="./pytorch" # Define the target directory where you want to apply the patch


# Apply the patch to the CMakeList in the target directory
patch "$TARGET_DIR/CMakeLists.txt" < "$PATCH_FILE_1"

# Apply the patch to the caffe2/CMakeLists.txt in the target directory
patch "$TARGET_DIR/caffe2/CMakeLists.txt" < "$PATCH_FILE_2"
