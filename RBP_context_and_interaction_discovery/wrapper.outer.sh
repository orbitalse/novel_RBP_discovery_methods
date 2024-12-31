#!/bin/bash

# Shaimae Elhajjajy
# May 11, 2023
# Run models for all RBPs (with CV, with hyperparameter tuning, with relabeling)

# Read from command line
if [ $# != 2 ]
then
  echo "Enter the cell line {HepG2, K562} and the number of iterations for hyperparameter tuning."
  echo "Important Note! Make sure to update the SBATCH array limits based on the cell line!"
  exit
fi

# Set variables
cell_line=$1
num_tuning_iterations=$2

# Submit jobs for each iteration of hyperparameter tuning
loop_limit=( $( expr $num_tuning_iterations - 1 ) )
for i in $(seq 0 $loop_limit)
do
  sbatch wrapper.inner.sh $cell_line $i
done


