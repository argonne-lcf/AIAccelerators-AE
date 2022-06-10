#!/bin/bash

SRC="all"
CACHE=./cache/${SRC} # Change it to valid cache directory
H5=/grand/projects/datascience/memani/uno-dataset/All.h5  # Change it to a valid H5 path
BATCH=4096             # Number of samples stored in one tfrecord file
TFR=${SRC}_TFR	       # Directory containing TF Records

python uno_to_tfrecords.py --train_sources $SRC --cache $CACHE --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z $BATCH --use_exported_data $H5 --export_data $TFR --on_memory_loader False
