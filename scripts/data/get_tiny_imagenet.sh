#!/bin/bash

# Specify the data path
# Example usage:
# . tiny-imagenet.sh /net/projects/yuxinlab/data/ tiny_imagenet)

# echo ${root}
root="${1}"
filename="${2}"
current=${root}${filename}
echo ${current}

# Download and unzip dataset
cd ${root}
pwd
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
# This will automatically create a folder called tiny-imagenet-200
mv tiny-imagenet-200/ ${filename}/

# Training data
cd $current/train
for DIR in $(ls); do (cd $DIR && rm *.txt && mv images/* . && rm -r images); done

# Validation data
cd $current/val
annotate_file="val_annotations.txt"

for i in $(seq 1 $length); do line=$(sed -n ${i}p $annotate_file); file=$(echo $line | cut -f1 -d" "); directory=$(echo $line | cut -f2 -d" "); mkdir -p $directory; mv images/$file $directory; done

rm -r images
echo "done"