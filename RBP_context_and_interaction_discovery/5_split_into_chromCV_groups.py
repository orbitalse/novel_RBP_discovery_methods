# Author: Shaimae Elhajjajy
# Date: January 7, 2022
# Purpose: Combine the positive and negative sequences into a single data frame and implement chromCV fold assignment

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Assign the regions (centered, extended eCLIP peaks) to different groups for cross-validation, based on the method used in BENGI.
# (For each BENGI dataset, we calculated the number of cCRE-gene pairs on each chromosome and assigned 
# chromCV groups accordingly. The chromosome with the most pairs (often chr1) was assigned its own group. 
# Then, we iteratively took the chromosome with the most and fewest pairs and combined them to create one CV group. 
# In total, the 23 chromosomes (1–22, X) were assigned to 12 CV groups.)

# ------------------------------------------------------ SET UP -------------------------------------------------------

# Load packages
import numpy as np
import pandas as pd
import sys

# Read in the data
if len(sys.argv) != 3:
    sys.exit("Enter the files containing the positive and negative sequences.")

# Set variables from command line
positive_seq_file = sys.argv[1]
negative_seq_file = sys.argv[2]

# ----------------------------------------------------- FUNCTIONS -----------------------------------------------------

# Function to fill the dictionary with number of regions in each chromosome
def get_num_regions_per_chr(region):
    chrom_dict[region.chr] += 1

# Function to create the CV groups based on the number of regions in each chromosome
# Note: much of this function is adapted from Jill's assign.groups.py BENGI script
def create_CV_groups(key_list):
    group_assignments = {}
    group_assignments[key_list[0]] = "CV-0"
    for i in range(1,12):
        group_assignments[key_list[i]] = "CV-" + str(i)
        group_assignments[key_list[0 - i]] = "CV-" + str(i)
    return(group_assignments)

# Function to assign each region to the appropriate CV group
def assign_to_CV_group(region):
    CV_group.append(group_assignments[region.chr])

# ------------------------------------------------------- MAIN --------------------------------------------------------

positive_seq_df = pd.read_csv(positive_seq_file, sep =  "\t", header = None)
negative_seq_df = pd.read_csv(negative_seq_file, sep =  "\t", header = None)

# Add the y labels for the positive sequences before pooling
positive_y_label = np.repeat("bound", positive_seq_df.shape[0])
positive_seq_df.insert(positive_seq_df.shape[1], "y_label", positive_y_label)

# Add the y labels for the negative sequences before pooling
negative_y_label = np.repeat("unbound", negative_seq_df.shape[0])
negative_seq_df.insert(negative_seq_df.shape[1], "y_label", negative_y_label)

# Create a data frame containing all training sequences
all_seq_df = pd.concat([positive_seq_df, negative_seq_df])
all_seq_df = all_seq_df.reset_index(drop = True)

# Randomly shuffle the samples
all_seq_df = all_seq_df.sample(frac = 1, axis = 0, random_state = 42).reset_index(drop = True)

# Rename the necessary columns
all_seq_df.rename(columns = {0: "chr", 1: "start", 2: "end", 5: "strand"}, inplace = True)

# Set up the dictionary that will contain the number of regions in each chromosome
key_list = list(range(1, 23))
key_list = ["chr" + str(num) for num in key_list]
key_list.append("chrX")
chrom_dict = {key: 0 for key in key_list}

# Calculate how many regions belong to each chromosome
for row in all_seq_df.itertuples():
    get_num_regions_per_chr(row)

# Sort the dictionary in descending order (this line of code is from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value)
sorted_chrom_dict = {k: v for k, v in sorted(chrom_dict.items(), key = lambda item: item[1], reverse = True)}

# Determine composition of CV groups, based on number of regions in each chr
group_assignments = create_CV_groups(list(sorted_chrom_dict.keys()))

# Create an empty list that will contain the CV group for each region
CV_group = list()

# Assign each region to its corresponding CV group
for row in all_seq_df.itertuples():
    assign_to_CV_group(row)

# Add the CV group to the data frame
all_seq_df.insert(all_seq_df.shape[1], "CV_group", CV_group)

# Write the positive regions to a file
all_seq_cv_groups_filename = positive_seq_file.split(".positive.seq.bed")[0] + ".all.training.seq.bed"
all_seq_df.to_csv("./" + all_seq_cv_groups_filename, sep = "\t", header = False, index = None, na_rep = "NULL")

# Calculate the size of each CV group (to make sure they are roughly the same size)
CV_group_chr = list()
CV_group_label = list()
CV_group_size = list()
for key in group_assignments:
    CV_group_chr.append(key)
    CV_group_label.append(group_assignments[key])
    CV_group_size.append(all_seq_df[all_seq_df.chr == key].shape[0])

# Create a data frame of the CV group sizes and write to a file
cv_groups_df = pd.DataFrame({"chr": CV_group_chr, "CV_group": CV_group_label, "size": CV_group_size})
cv_groups_filename = positive_seq_file.split(".positive.seq.bed")[0] + ".chromCVgroups.all.tsv"
cv_groups_df.to_csv("./" + cv_groups_filename, sep = "\t", header = False, index = None)




