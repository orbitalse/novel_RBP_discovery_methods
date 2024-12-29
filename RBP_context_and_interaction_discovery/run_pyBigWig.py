# Author: Shaimae Elhajjajy
# Date: January 7, 2022
# Purpose: Extract the signal over a particular region of interest (e.g., peak, kmer)

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# ------------------------------------------------------ SET UP -------------------------------------------------------

# Load packages
import pandas as pd
import pyBigWig
import sys

# Read in the data
if len(sys.argv) != 6:
    sys.exit("Enter files containing:\n(1) the regions of interest in BED format\n\
(2) eCLIP plus strand signal\n(3) eCLIP minus strand signal\n(4) the signal type (mean, max, min, coverage, std)\n(5) the output file name.")

# Set variables from command line
regions_file = sys.argv[1]
eCLIP_plus_signal_file = sys.argv[2]
eCLIP_minus_signal_file = sys.argv[3]
signal_type = sys.argv[4]
outfile = sys.argv[5]

# ----------------------------------------------------- FUNCTIONS -----------------------------------------------------

# Function to calculate the eCLIP signal over the region of interest
def compute_eCLIP_signal_over_region(region, eCLIP_plus_signal_file, eCLIP_minus_signal_file, signal_type):
    if (region.strand == "+"):
        bw = pyBigWig.open(eCLIP_plus_signal_file)
        signal_over_region = bw.stats(region.chr, region.start, region.end, type = signal_type)
    elif (region.strand == "-"):
        bw = pyBigWig.open(eCLIP_minus_signal_file)
        # Flip min and max to account for negative signal on the minus strand
        if (signal_type == "max"):
            signal_over_region = bw.stats(region.chr, region.start, region.end, type = "min")
        elif (signal_type == "min"):
            signal_over_region = bw.stats(region.chr, region.start, region.end, type = "max")
        else:
            signal_over_region = bw.stats(region.chr, region.start, region.end, type = signal_type)

    signal_over_region = signal_over_region[0]

    return(signal_over_region)

# ------------------------------------------------------- MAIN --------------------------------------------------------

# Read in the regions of interest
regions_df = pd.read_csv(regions_file, sep = "\t", header = None)
regions_df.rename(columns = {0: "chr", 1: "start", 2: "end", 5: "strand"}, inplace = True)

# Create an empty list to contain the eCLIP signal over each region of interest
signal_list = list()

# Compute the eCLIP signal over each region
for row in regions_df.itertuples():
    signal_over_region = compute_eCLIP_signal_over_region(row, eCLIP_plus_signal_file, eCLIP_minus_signal_file, signal_type)
    signal_list.append(signal_over_region)

# Add the signal to the data frame
regions_df.insert(regions_df.shape[1], "signal_" + signal_type, signal_list)

# Write the data frame to a file
regions_df.to_csv("./" + outfile, sep = "\t", header = False, index = None, na_rep = "NULL")



