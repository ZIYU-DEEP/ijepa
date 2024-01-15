#!/bin/bash

# Specify the data path
# Example usage:
# . get-imagenet.sh /data/ imagenet_1k

# echo ${root}
root="${1}"
folder_name="${2}"
current=${root}${folder_name}
echo ${current}

# Get the current folder
cd ${current}
mkdir train && mkdir val
pwd

# Download the dataset if you haven't
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate

# Process for the training set
mv ILSVRC2012_img_train.tar train/
cd train
tar -xvf ILSVRC2012_img_train.tar 
rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# Process for the validation set
mv ILSVRC2012_img_val.tar val/ 
cd val
tar -xvf ILSVRC2012_img_val.tar
rm -f ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..

# Check the total files
find train/ -name "*.JPEG" | wc -l
find val/ -name "*.JPEG" | wc -l
