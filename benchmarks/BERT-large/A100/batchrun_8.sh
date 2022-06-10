#!/bin/bash -l
cd ./BERT
module load conda/tensorflow
conda activate
#results/phase_1/
scripts/run_pretraining_lamb_phase1.sh 60 10 8 "7.5e-4" "5e-4" "fp16" "true" 8 "2133" "213" 400 100 128 384 "large" > rplp1.400steps_8_A100.out 2>&1

#train_batch_size_phase1=${1:-60}
#train_batch_size_phase2=${2:-10}
#eval_batch_size=${3:-8}
#learning_rate_phase1=${4:-"7.5e-4"}
#learning_rate_phase2=${5:-"5e-4"}
#precision=${6:-"fp16"}
#use_xla=${7:-"true"}
#num_gpus=${8:-1}
#warmup_steps_phase1=${9:-"2133"}
#warmup_steps_phase2=${10:-"213"}
#train_steps=${11:-400}
#save_checkpoints_steps=${12:-100}
#num_accumulation_steps_phase1=${13:-128}
#num_accumulation_steps_phase2=${14:-384}
#bert_model=${15:-"large"}

# 1 gpu
#60 10 8 "7.5e-4" "5e-4" "fp16" "true" 1 "2133" "213" 400 100 128 384 "large"
# 8 gpus
#60 10 8 "7.5e-4" "5e-4" "fp16" "true" 8 "2133" "213" 400 100 128 384 "large"

