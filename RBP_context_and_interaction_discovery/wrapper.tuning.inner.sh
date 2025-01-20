#!/bin/bash

#SBATCH --nodes 1
#SBATCH --time=120:00:00
#SBATCH --mem=200G
#SBATCH --job-name=x
#SBATCH --output=/home/elhajjajys/slurm_logs/jobid_%A_%a.output
#SBATCH --error=/home/elhajjajys/slurm_logs/jobid_%A_%a.error
#SBATCH --array=[0-102]%20
#SBATCH --partition=5days

# Shaimae Elhajjajy
# May 11, 2023
# Run models for all RBPs (with CV, with hyperparameter tuning, with relabeling)

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Read from command line
if [ $# != 2 ]
then
  echo "Enter the cell line {HepG2, K562} and the iteration number."
  echo "Important Note! Make sure to update the SBATCH array limits based on the cell line!"
  exit
fi

# Set variables
cell_line=$1
tuning_iteration_num=$2

baseline_kmer_size_list=5
baseline_step_size_list=1
context_target_size=5
context_target_step=1
context_kmer_size=5
context_kmer_step=5
num_contexts=5
threshold=0.5
num_iterations=3

# Set working directories
MAIN_DIR=/data/zusers/elhajjajys/RBP_prediction
DATA_DIR=$MAIN_DIR/training_sequences/eCLIP_only/results/ext_50/$cell_line
WORKING_DIR=/data/zusers/elhajjajys/RBP_prediction/nlp/method2/baseline_on_contexts/results_allRBPs_tuning_denseEmbed_051123
OUT_DIR=$WORKING_DIR/$cell_line
DOCKER=/home/elhajjajys/bin/python3-R.simg

# Get list of RBPs
RBPs=( $( ls $DATA_DIR ) )

# Get RBP for this job
RBP=${RBPs[$SLURM_ARRAY_TASK_ID]}
echo $RBP

# Create tmp dir
mkdir -p /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID
cd /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID

# Random sleep
MINWAIT=10
MAXWAIT=120
sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))

# Copy files to tmp dir
cp $WORKING_DIR/*.py ./
cp $DATA_DIR/$RBP/$RBP*.all.training.seq.bed ./

all_seq_file=`ls ./$RBP*.all.training.seq.bed`

echo "Running for $RBP in $cell_type..."
echo "--> baseline_kmer_size_list = $baseline_kmer_size_list"
echo "--> baseline_step_size_list = $baseline_step_size_list"
echo "--> context_target_size = $context_target_size"
echo "--> context_target_step = $context_target_step"
echo "--> context_kmer_size = $context_kmer_size"
echo "--> context_kmer_step = $context_kmer_step"
echo "--> num_contexts = $num_contexts"
echo "--> threshold = $threshold"
echo "--> num_iterations = $num_iterations"
echo "--> baseline_model_type = dense"
echo "--> baseline_embedding_type = BoW"
echo "--> contexts_model_type = dense_embed"
echo "--> contexts_embedding_type = embedding"

# Run model
singularity exec $DOCKER python3 baseline_on_contexts.tuning.py $all_seq_file \
                                                                $baseline_kmer_size_list \
                                                                $baseline_step_size_list \
                                                                $context_target_size \
                                                                $context_target_step \
                                                                $context_kmer_size \
                                                                $context_kmer_step \
                                                                $num_contexts \
                                                                $threshold \
                                                                $num_iterations

# Create output directory
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/$RBP
mkdir -p $OUT_DIR/$RBP/iteration$tuning_iteration_num

# Random sleep
MINWAIT=10
MAXWAIT=120
sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))

# Copy results to /data
cp *.tsv $OUT_DIR/$RBP/iteration$tuning_iteration_num
cp *.fa $OUT_DIR/$RBP/iteration$tuning_iteration_num
cp *.obj $OUT_DIR/$RBP/iteration$tuning_iteration_num

# Clean up
cd $OUT_DIR
rm -r /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID












