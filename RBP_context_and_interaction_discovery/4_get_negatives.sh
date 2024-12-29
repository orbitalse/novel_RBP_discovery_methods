#!/bin/bash

# Author: Shaimae Elhajjajy
# Date: March 10, 2022
# Purpose: Process randomly sampled matched genomic regions to generate negative sequences for training

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Read in from command line
if [ $# != 3 ]
then
  echo "Enter the file containing the randomly sampled matched genomic regions, the RBP, and the cell type."
  exit
fi

# Set variables
genomic_sample=$1
RBP=$2
cell_type=$3

out_file_ID=`basename $genomic_sample ".genomic.sample.extended.bed"`

# Set directories
DOCKER=/home/elhajjajys/bin/python3-R.simg

# Extract sequences for the centered, extended regions (Note: this is strand-specific)
bedtools getfasta -s -fi GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -bed $genomic_sample -bedOut > $out_file_ID.genomic.sample.seq.tmp

# Keep only the regions that contain A, C, G, T (remove sequences that contain other nucleotide symbols, like K, N, W, Y, etc.)
awk '{FS=OFS="\t"}{if ($9 ~ /^[ACGT]+$/) print $0}' $out_file_ID.genomic.sample.seq.tmp > $out_file_ID.genomic.sample.seq.bed

# Compute the average, stdev, minimum, maximum, and coverage of eCLIP signal over the negative sequences
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.genomic.sample.seq.bed $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig mean $out_file_ID.genomic.sample.signal.mean.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.genomic.sample.signal.mean.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig std $out_file_ID.genomic.sample.signal.std.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.genomic.sample.signal.std.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig min $out_file_ID.genomic.sample.signal.min.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.genomic.sample.signal.min.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig max $out_file_ID.genomic.sample.signal.max.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.genomic.sample.signal.max.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig coverage $out_file_ID.genomic.sample.signal.coverage.tmp

mv $out_file_ID.genomic.sample.signal.coverage.tmp $out_file_ID.negative.seq.bed


