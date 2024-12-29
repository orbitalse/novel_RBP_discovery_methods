#!/bin/bash

# Author: Shaimae Elhajjajy
# Date: March 3, 2022
# Purpose: Process eCLIP peaks.

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Read from command line
if [ $# != 3 ]
then
  echo "Enter the RBP, the cell type, and the output file identifier."
  exit
fi

# Set variables
RBP=$1
cell_type=$2
out_file_ID=$3

peaks=$out_file_ID.all.bed

# Set directories
DOCKER=/home/elhajjajys/bin/python3-R.simg

# Add peak length
awk '{print $0 "\t" ($3-$2)}' $peaks > $out_file_ID.peaks.length.tmp

# Extract sequences for the peaks (Note: this is strand-specific)
bedtools getfasta -s -fi GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -bed $out_file_ID.peaks.length.tmp -bedOut > $out_file_ID.peaks.seq.tmp

# Compute the average, stdev, minimum, maximum, and coverage of eCLIP signal over the peaks
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.peaks.seq.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig mean $out_file_ID.peaks.signal.mean.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.peaks.signal.mean.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig std $out_file_ID.peaks.signal.std.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.peaks.signal.std.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig min $out_file_ID.peaks.signal.min.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.peaks.signal.min.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig max $out_file_ID.peaks.signal.max.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.peaks.signal.max.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig coverage $out_file_ID.peaks.signal.coverage.tmp

mv $out_file_ID.peaks.signal.coverage.tmp $out_file_ID.peaks.seq.bed




