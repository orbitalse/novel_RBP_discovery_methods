#!/bin/bash

# Shaimae Elhajjajy
# June 9, 2023
# Run models for RBPs (with CV, hyperparameter tuning, relabeling, fitting a new model)
# Note: modified 04/19/24 to run on all chromCV groups.

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Read from command line
if [ $# != 2 ]
then
  echo "Enter the cell line {HepG2, K562} and the RBP."
  exit
fi
# For this job, array limit will be the # of RBPs with baseline/contexts results (some models did not train successfully).
# i.e., wc -l best_tuning_iterations_per_RBP.tsv (-1 to account for header)

# Set variables
cell_line=$1
RBP=$2

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
WORKING_DIR=$MAIN_DIR/nlp/method2/baseline_on_contexts/results_allRBPs_tuning_denseEmbed_051123
OUT_DIR=$WORKING_DIR/results_pooled_chromCVgroups_041524
DOCKER=/zata/zippy/elhajjajys/bin/python3-r.simg

# Create tmp dir
mkdir -p /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID
cd /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID

# Get list of best iterations for each RBP
cp $WORKING_DIR/$cell_line/best_tuning_iterations_per_RBP.tsv .

# Get best baseline and context iterations
baseline_iteration=( $( awk -v RBP=$RBP '{if ($1 == RBP) print $2}' best_tuning_iterations_per_RBP.tsv ) )
contexts_iteration=( $( awk -v RBP=$RBP '{if ($1 == RBP) print $5}' best_tuning_iterations_per_RBP.tsv ) )

# Get CV group for this job
cv_folds=("CV-0" "CV-1" "CV-2" "CV-3" "CV-4" "CV-5" "CV-6" "CV-7" "CV-8" "CV-9" "CV-10" "CV-11")
test_cv_fold=${cv_folds[$SLURM_ARRAY_TASK_ID]}

# Copy files to tmp dir
cp $WORKING_DIR/*.py ./
cp $DATA_DIR/$RBP/$test_cv_fold/$RBP*.bed ./
cp $WORKING_DIR/$cell_line/$RBP/$baseline_iteration/hyperparameters.tsv ./baseline_hyperparameters.tsv
cp $WORKING_DIR/$cell_line/$RBP/$contexts_iteration/hyperparameters.tsv ./contexts_hyperparameters.tsv

training_seq_file=`ls ./$RBP*.all.training.seq.training.bed`
test_seq_file=`ls ./$RBP*.all.training.seq.test.bed`

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
echo "--> embedding_type = $embedding_type"
echo "--> baseline_model_type = $baseline_model_type"
echo "--> contexts_model_type = $contexts_model_type"

# Run model
singularity exec $DOCKER python3 baseline_on_contexts.relabeling.py $training_seq_file \
                                                                    $test_seq_file \
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
OUT_DIR=$OUT_DIR/$cell_line/$RBP/relabeling_newfit/$test_cv_fold
mkdir -p $OUT_DIR

# Copy results to /data
cp *.tsv $OUT_DIR
cp *.fa $OUT_DIR
cp *.obj $OUT_DIR

# Clean up
cd $OUT_DIR
rm -r /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID








