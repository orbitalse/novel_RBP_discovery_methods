# Author: Shaimae Elhajjajy
# Date: March 3, 2022
# Purpose: Randomly sample sequences to create negative training examples.
# Requirements (balanced dataset)
# - sampled sequences must be from matched genomic regions as their analogous peaks
# - sampled sequences must be from the same chromosome as their analogous peaks
# - sampled sequences must be the same length as their analogous peaks

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

import numpy as np
import pandas as pd
import random
import sys

if len(sys.argv) != 4:
    sys.exit("Enter the input file (either positive sequences or overlaps), the output file, and the directory containing classified GTF files")

input_file = sys.argv[1]
output_file = sys.argv[2]
classified_gtfs = sys.argv[3]

GTF_DIR = "./" + classified_gtfs

# Skip unclassified peaks 
regions = ["tRNA", "miRNA", "miRNA.proximal", "CDS", "3UTR", "5UTR", "5SS", "3SS", "intron.proximal", "intron.distal", "exon.noncoding"]

# ----------------------------------------------------- FUNCTIONS -----------------------------------------------------

def load_sequences(filename):
    df = pd.read_csv(filename, sep =  "\t", header = None)
    if (df.shape[1] == 8): # overlaps
        df.columns = ["chr", "start", "end", "filler1", "filler2", "strand", "peakID", "region"]
    else: # positive sequences
        df.columns = ["chr", "start", "end", "label", "1000", "strand", "peakID", "region", "seq", "seq_signal_mean", \
                         "seq_signal_std", "seq_signal_min", "seq_signal_max", "seq_signal_coverage"]

    return(df)

def load_annotation(region_annotation_file, region):
    region_annot_df = pd.read_csv(region_annotation_file, sep =  "\t", header = None)
    if ((region == "tRNA") or (region == "miRNA") or (region == "miRNA.proximal")):
        region_annot_df.columns = ["chr", "start", "end", "filler1", "filler2", "strand", "geneID"]
    elif ((region == "CDS") or (region == "exon.noncoding")):
        region_annot_df.columns = ["chr", "start", "end", "filler1", "filler2", "strand", "geneID", "txptID"]
    elif ((region == "3UTR") or (region == "5UTR") or (region == "5SS") or (region == "3SS") or \
            (region == "intron.proximal") or (region == "intron.distal")):
        region_annot_df.columns = ["chr", "start", "end", "filler1", "filler2", "strand", "geneID", "txptID", "regionID"]

    return(region_annot_df)

def get_random_region(region_annot_df):
    random_region = region_annot_df.sample(1)
    
    return(random_region)

def get_random_coords(random_region):

    random_start = random.randint(int(random_region.start), int(random_region.end))
    random_end = random_start + 1

    return(random_start, random_end)

def get_random_sequence(region_annot_df):
    
    random_region = get_random_region(region_annot_df)
    random_start, random_end = get_random_coords(random_region)
    random_strand = random_region.strand.to_string(index = False)

    return(random_start, random_end, random_strand)

# ----------------------------------------------------- MAIN -----------------------------------------------------

df = load_sequences(input_file)

negative_seq_chrs = []
negative_seq_starts = []
negative_seq_ends = []
negative_seq_strands = []
negative_seq_peakIDs = []
negative_seq_regions = []

# For each region, do the following:
for region in regions:
    print("Sampling negative regions for " + region + "...")
    # Get all positive sequences in this region (they will be handled at the same time so the annotation file for the region
    # can be read into Python only once)
    region_seq_df = df[df["region"] == region].reset_index(drop = True)
    if (region_seq_df.shape[0] != 0):
        # Get a list of (unique) chromosomes for peaks in this region
        unique_chrs = list(set(region_seq_df["chr"]))
        unique_chrs.sort()
        # For every chromosome, do the following:
        for chr in unique_chrs:
            # Get all positive sequences in this region and chromosome
            chr_seq_df = region_seq_df[region_seq_df["chr"] == chr].reset_index(drop = True)
            # Read in the genome annotation file for this region and chromosome
            region_annot_df = load_annotation(GTF_DIR + "/" + region + "/" + chr + ".bed", region)
            # For every sequence in this region and chromosome, randomly select a negative region of the same size
            for row in chr_seq_df.itertuples():
                start, end, strand = get_random_sequence(region_annot_df)

                negative_seq_chrs.append(chr)
                negative_seq_starts.append(start)
                negative_seq_ends.append(end)
                negative_seq_strands.append(strand)
                negative_seq_peakIDs.append(row.peakID)
                negative_seq_regions.append(region)

filler1 = ["."] * len(negative_seq_chrs)
filler2 = filler1
negative_seq_df = pd.DataFrame({"chr": negative_seq_chrs, "start": negative_seq_starts, "end": negative_seq_ends, \
                                    "filler1": filler1, "filler2": filler2, "strand": negative_seq_strands, \
                                    "peakID": negative_seq_peakIDs, "region": negative_seq_regions})

negative_seq_df.to_csv(output_file, sep = "\t", index = False, header = False)





