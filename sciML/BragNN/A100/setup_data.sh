#!/bin/bash

# Download the dataset from Zhengchun's repo located at: https://github.com/lzhengchun/BraggNN

git_repo='https://github.com/lzhengchun/BraggNN'

echo "Downloading dataset located at $git_repo"

git clone $git_repo

#Move the dataset directory to the main and delete BraggNN
mv BraggNN/dataset .
rm -rf BraggNN

echo "Untar the dataset"

cd dataset
tar -xvf dataset.tar.gz
