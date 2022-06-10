#!/bin/bash -l
#COBALT -t 720
#COBALT -n 1
#COBALT -O /home/vsastry/$COBALT_JOBID

echo [$SECONDS] setup conda environment
module load conda/2021-11-30
conda activate /lus/theta-fs0/projects/datascience/vsastry/Benchmarks/Pilot1/Uno/uno_conda_env

echo [$SECONDS] python = $(which python)
echo [$SECONDS] python version = $(python --version)

#echo [$SECONDS] setup local env vars
NODES=`cat $COBALT_NODEFILE | wc -l`
RANKS_PER_NODE=1
RANKS=$((NODES * RANKS_PER_NODE))
echo [$SECONDS] NODES=$NODES  RANKS_PER_NODE=$RANKS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=64

echo [$SECONDS] run convert to tfrecords
#python ilsvrc_dataset_serial.py -c ilsvrc.json --logdir logdir/${COBALT_JOBID}-serial
cd /lus/theta-fs0/projects/datascience/vsastry/UNO/from_gc/uno/Uno

#python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 4096 --export_data All.h5 --shuffle True  
#python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 2048 --use_exported_data /grand/projects/datascience/memani/uno-dataset/All.h5 --cp True --shuffle True --tb True 
python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 512 --use_exported_data /grand/projects/datascience/memani/uno-dataset/All.h5  --use_tfrecords all_TFR --cp True --shuffle True --tb True
echo [$SECONDS] done
