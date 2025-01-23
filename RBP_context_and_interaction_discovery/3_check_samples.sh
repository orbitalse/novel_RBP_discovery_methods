#!/bin/bash

# Author: Shaimae Elhajjajy
# Date: March 9, 2022
# Purpose: After first round of sampling from matched genomic regions, resample as needed

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Read from command line
if [ $# != 3 ]
then
  echo "Enter the file containing sampled regions, the file containing positive sequences, and the extension length."
  exit
fi

# Set variables
sampled_regions=$1
positive_seq=$2
extension=$3

# Set directories
GENOMES=./genomes

out_file_ID=`basename $sampled_regions ".bed"`

# Extend the centered peak by the same amount in either direction
bedtools slop -i $sampled_regions -g $GENOMES/GRCh38_EBV.chrom.sizes.ENCODE.ref.tsv -b $extension > regions.extended.tmp

# Positive and negative regions should not overlap - resample if needed
bedtools intersect -s -u -a regions.extended.tmp -b $positive_seq > overlaps.tmp

while [ -s overlaps.tmp ]
do
  echo "Some sampled genomic regions overlap with positive regions. Resampling..."
  # Remove regions that overlap
  comm -23 <(sort regions.extended.tmp) <(sort overlaps.tmp) > regions.extended.bed
  # Resample regions to replace those that overlap
  python3 2_sample_genomic_regions.py overlaps.tmp resampled.tmp gencodeV29
  # Extend resampled regions to matching size
  bedtools slop -i resampled.tmp -g $GENOMES/GRCh38_EBV.chrom.sizes.ENCODE.ref.tsv -b $extension > resampled.extended.tmp
  # Add resampled regions to the main list
  cat regions.extended.bed resampled.extended.tmp > regions.extended.tmp
  # Check again for overlap
  bedtools intersect -s -u -a regions.extended.tmp -b $positive_seq > overlaps.tmp
done

mv regions.extended.tmp $out_file_ID.extended.bed

rm -f $sampled_regions
rm -f overlaps.tmp
rm -f regions.extended.bed
rm -f resampled.tmp
rm -f resampled.extended.tmp




