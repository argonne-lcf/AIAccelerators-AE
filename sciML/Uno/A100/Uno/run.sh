#!/bin/bash
python uno_baseline_keras2.py --train_sources all --cache cache/all \
--use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True \
-z 256 --use_exported_data /grand/projects/datascience/memani/uno-dataset/All.h5  --use_tfrecords all_TFR --cp True --shuffle True --tb True
