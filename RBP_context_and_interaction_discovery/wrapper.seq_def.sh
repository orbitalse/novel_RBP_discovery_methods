#!/bin/bash

#SBATCH --nodes 1
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --job-name=seq
#SBATCH --output=/home/elhajjajys/slurm_logs/jobid_%A_%a.output
#SBATCH --error=/home/elhajjajys/slurm_logs/jobid_%A_%a.error
#SBATCH --array=[0-102]%30
#SBATCH --partition=4hours

# Author: Shaimae Elhajjajy
# Date: March 10, 2022
# Purpose: Get training sequences for all RBPs.

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Read from command line
if [ $# != 3 ]
then
  echo "Enter the metadata file for RBP eCLIP peaks (1st column: expID, 2nd column: fileID, 3rd column: RBP), the genome annotation (e.g., gencodeV29), and the extension length."
  exit
fi

## Set variables
metadata=$1
genome_annot=$2
extension=$3

## Set directories
METADATA=/home/elhajjajys/metadata
DOCKER=/home/elhajjajys/bin/python3-R.simg
GENOMES=./genomes
MAIN_DIR=./RBP_prediction/training_sequences/eCLIP_only
SCRIPT_DIR=$MAIN_DIR/scripts
OUT_DIR=$MAIN_DIR/results/ext_$extension
CLASSIFIED_PEAKS_DIR=./classified_by_genomic_region/eCLIP_peaks/$genome_annot
GTF_DIR=./classified_by_genomic_region/gtfs/$genome_annot
BIGWIGS=./eCLIP_data/merged_bigWigs

# Extract metadata
numExps=$( wc -l $METADATA/$metadata | cut -f 1 -d " " | awk '{print $1 - 1}' )
expIDs=( $( awk '{print $1}' $METADATA/$metadata ) )
expID=${expIDs[$SLURM_ARRAY_TASK_ID]}
fileIDs=( $( awk '{print $2}' $METADATA/$metadata ) )
fileID=${fileIDs[$SLURM_ARRAY_TASK_ID]}
RBPs=( $( awk '{print $3}' $METADATA/$metadata ) )
RBP=${RBPs[$SLURM_ARRAY_TASK_ID]}
cellType=$( echo $metadata | cut -f 3 -d "_" | cut -f 1 -d "." )

echo "Setting up..."

# Create the (outer) output directory
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/$cellType
mkdir -p $OUT_DIR/$cellType/$RBP

# Create a temporary directory for computation in /tmp
mkdir -p /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID
cd /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID

# Create the output file identifiers (all output files will have this name)
out_file_ID=$RBP"_"$cellType"_"$expID"_"$fileID

# Copy over the necessary files from /data
cp $CLASSIFIED_PEAKS_DIR/$cellType/$RBP/$out_file_ID.all.bed ./
cp $GENOMES/GRCh38_EBV.chrom.sizes.ENCODE.ref.tsv ./
cp $GENOMES/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta ./
cp $BIGWIGS/$cellType/$RBP"_"* ./
cp -r $GTF_DIR ./

# Copy over the necessary scripts from /data
cp $SCRIPT_DIR/0_process_peaks.sh ./
cp $SCRIPT_DIR/1_get_positives.sh ./
cp $SCRIPT_DIR/2_sample_genomic_regions.py ./
cp $SCRIPT_DIR/3_check_samples.sh ./
cp $SCRIPT_DIR/4_get_negatives.sh ./
cp $SCRIPT_DIR/5_split_into_chromCV_groups.py ./
cp $SCRIPT_DIR/run_pyBigWig.py ./

# Print the RBP that is being processed by the job
echo "Generating training sequences for $RBP ($cellType)..."

# Run the pipeline for all RBPs
./0_process_peaks.sh $RBP $cellType $out_file_ID
./1_get_positives.sh $RBP $cellType $out_file_ID $extension
singularity exec $DOCKER python3 2_sample_genomic_regions.py $out_file_ID.positive.seq.bed $out_file_ID.genomic.sample.bed $genome_annot
./3_check_samples.sh $out_file_ID.genomic.sample.bed $out_file_ID.positive.seq.bed $extension
./4_get_negatives.sh $out_file_ID.genomic.sample.extended.bed $RBP $cellType 

# Check that positive and negative samples have the same amount;
# If not, it means some were removed b/c sequences contained non-ACGT nucleotides
# Repeat the process until they become equal.
while [ "$(wc -l < $out_file_ID.positive.seq.bed)" -ne "$(wc -l < $out_file_ID.negative.seq.bed)" ]
do
  echo "Positive and negative samples are not present at the same frequency. Resampling..."
  singularity exec $DOCKER python3 2_sample_genomic_regions.py $out_file_ID.positive.seq.bed $out_file_ID.genomic.sample.bed $genome_annot
  ./3_check_samples.sh $out_file_ID.genomic.sample.bed $out_file_ID.positive.seq.bed $extension
  ./4_get_negatives.sh $out_file_ID.genomic.sample.extended.bed $RBP $cellType
done

singularity exec $DOCKER python3 5_split_into_chromCV_groups.py $out_file_ID.positive.seq.bed $out_file_ID.negative.seq.bed

# Remove temporary files
rm *.tmp
rm $out_file_ID.all.bed

# Move output to the correct directory
mv $out_file_ID* $OUT_DIR/$cellType/$RBP

# Remove the temporary directory
cd $OUT_DIR/$cellType/$RBP
rm -r /tmp/elhajjajys/$SLURM_JOBID-$SLURM_ARRAY_TASK_ID

echo "Done."







