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
  echo "Enter the cell line {HepG2, K562} and the metadata file."
  exit
fi

# Set variables
cell_line=$1
metadata=$2

# Set working directories
MAIN_DIR=./RBP_prediction
WORKING_DIR=$MAIN_DIR/nlp/method2/baseline_on_contexts/results_allRBPs_tuning_denseEmbed_051123

LOG_DIR=./slurm_logs/RBP_prediction/oldcode_chromCV_relabel_041524/$cell_line/relabeling_newfit
mkdir -p $LOG_DIR

# Get number of RBPs
#RBPs=(RBFOX2)
RBPs=( $( awk 'NR>1{print $1}' $WORKING_DIR/$cell_line/best_tuning_iterations_per_RBP.tsv ) )
num_RBPs=${#RBPs[@]}

# Get loop limit
loop_limit=( $( expr $num_RBPs - 1 ) )

cv_folds=("CV-0" "CV-1" "CV-2" "CV-3" "CV-4" "CV-5" "CV-6" "CV-7" "CV-8" "CV-9" "CV-10" "CV-11")
num_cv_folds=${#cv_folds[@]}
array_limit=( $( expr $num_cv_folds - 1 ) )

# For loop - number of RBPs
for i in $(seq 0 $loop_limit)
do
  RBP=${RBPs[$i]}
  # Set memory and partition based on dataset size
  mem=( $( awk -v RBP=$RBP '{FS=OFS="\t"}{if ($2 == RBP) print $4}' $metadata ) )
  partition=( $( awk -v RBP=$RBP '{FS=OFS="\t"}{if ($2 == RBP) print $5}' $metadata ) )
  time=( $( awk -v RBP=$RBP '{FS=OFS="\t"}{if ($2 == RBP) print $6}' $metadata ) )
  # Array job, one per iteration (hyperparameter tuning) or CV fold (relabeling)
  job_name=run$i
  mkdir -p $LOG_DIR/$RBP
  sbatch --nodes=1 --job-name=$job_name --mem=$mem --exclude=z1020 \
          --time=$time --partition=$partition \
          --output=$LOG_DIR/$RBP/$RBP"_CV-"%a.output \
          --error=$LOG_DIR/$RBP/$RBP"_CV-"%a.error \
          --array=[0-$array_limit]%20 \
          wrapper.relabeling.inner.sh $cell_line $RBP
done



