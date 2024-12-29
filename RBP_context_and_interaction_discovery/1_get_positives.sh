#!/bin/bash

# Author: Shaimae Elhajjajys
# Date: March 10, 2022
# Purpose: Set up positive training sequences by centering and extending eCLIP peaks

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Read from command line
if [ $# != 4 ]
then
  echo "Enter the RBP, the cell type, the output file identifier, and the extension length."
  exit
fi

# Set variables
RBP=$1
cell_type=$2
out_file_ID=$3
extension=$4

# Set directories
DOCKER=/home/elhajjajys/bin/python3-R.simg

# Find the center of the peaks
awk '{FS=OFS="\t"}{center=int($2+(($3-$2)/2)); print $1, center, center + 1, $4, $5, $6, $11, $12}' $out_file_ID.peaks.seq.bed > $out_file_ID.peaks.centered.tmp

# Extend the centered peak by the same amount in either direction
bedtools slop -i $out_file_ID.peaks.centered.tmp -g GRCh38_EBV.chrom.sizes.ENCODE.ref.tsv -b $extension > $out_file_ID.peaks.extended.tmp

# Extract sequences for the centered, extended peaks (Note: this is strand-specific)
bedtools getfasta -s -fi GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -bed $out_file_ID.peaks.extended.tmp -bedOut > $out_file_ID.positive.seq.tmp

# Remove unclassified peaks (i.e., not assigned to a genomic region)
grep -v unclassified $out_file_ID.positive.seq.tmp > $out_file_ID.positive.seq.regions.tmp

# Keep only the regions that contain A, C, G, T (remove sequences that contain other nucleotide symbols, like K, N, W, Y, etc.)
# Remove all regions coming from chromosomes not in the main 23 chromosomes(1-22, X), such as chrY or chrM (among others)
grep -wE 'chr1|chr2|chr3|chr4|chr5|chr6|chr7|chr8|chr9|chr10|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr21|chr22|chrX' $out_file_ID.positive.seq.regions.tmp | awk '{FS=OFS="\t"}{if ($9 ~ /^[ACGT]+$/) print $0}' > $out_file_ID.positive.seq.filtered.tmp

# Compute the average, stdev, minimum, maximum, and coverage of eCLIP signal over the positive sequences
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.positive.seq.filtered.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig mean $out_file_ID.positive.seq.signal.mean.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.positive.seq.signal.mean.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig std $out_file_ID.positive.seq.signal.std.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.positive.seq.signal.std.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig min $out_file_ID.positive.seq.signal.min.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.positive.seq.signal.min.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig max $out_file_ID.positive.seq.signal.max.tmp
singularity exec $DOCKER python3 run_pyBigWig.py $out_file_ID.positive.seq.signal.max.tmp $RBP*.merged_signal_plus.*.bigWig $RBP*.merged_signal_minus.*.bigWig coverage $out_file_ID.positive.seq.signal.coverage.tmp

mv $out_file_ID.positive.seq.signal.coverage.tmp $out_file_ID.positive.seq.bed




