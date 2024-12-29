# Shaimae Elhajjajy
# July 13, 2022
# Define Preprocessor class for import to other scripts

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

import contexts
import encoder 
import numpy as np
import pandas as pd
import pickle
import random
import re
from scipy.stats import chi2_contingency
import os

class Preprocessor:
    def __init__(self):
        self.seq_objects = [] 
        self.split_seq_as_strings = [] 
        self.split_seq_labels = [] 
        self.split_seq_ids = [] 
        self.split_seq_cv_groups = []
        self.contexts_as_strings = [] 
        self.context_labels = [] 
        self.context_seq_ids = []
        self.context_ids = []
        self.context_cv_groups = []

        self.seq_fasta_headers_all = []
        self.seq_fasta_seq_all = []
        self.seq_fasta_headers_true_positive = []
        self.seq_fasta_seq_true_positive = []
        self.seq_fasta_headers_true_negative = []
        self.seq_fasta_seq_true_negative = []
        self.seq_fasta_headers_predicted_positive = []
        self.seq_fasta_seq_predicted_positive = []
        self.seq_fasta_headers_predicted_negative = []
        self.seq_fasta_seq_predicted_negative = []

        self.ctxt_fasta_headers_all = []
        self.ctxt_fasta_seq_all = []
        self.ctxt_fasta_headers_true_positive = []
        self.ctxt_fasta_seq_true_positive = []
        self.ctxt_fasta_headers_true_negative = []
        self.ctxt_fasta_seq_true_negative = []
        self.ctxt_fasta_headers_predicted_positive = []
        self.ctxt_fasta_seq_predicted_positive = []
        self.ctxt_fasta_headers_predicted_negative = []
        self.ctxt_fasta_seq_predicted_negative = []

        self.baseline_on_ctxt_fasta_headers_all = []
        self.baseline_on_ctxt_fasta_seq_all = []
        self.baseline_on_ctxt_fasta_headers_true_positive = []
        self.baseline_on_ctxt_fasta_seq_true_positive = []
        self.baseline_on_ctxt_fasta_headers_true_negative = []
        self.baseline_on_ctxt_fasta_seq_true_negative = []
        self.baseline_on_ctxt_fasta_headers_predicted_positive = []
        self.baseline_on_ctxt_fasta_seq_predicted_positive = []
        self.baseline_on_ctxt_fasta_headers_predicted_negative = []
        self.baseline_on_ctxt_fasta_seq_predicted_negative = []

        self.ctxt_relabeled_fasta_headers_all = []
        self.ctxt_relabeled_fasta_seq_all = []
        self.ctxt_relabeled_fasta_headers_true_positive = []
        self.ctxt_relabeled_fasta_seq_true_positive = []
        self.ctxt_relabeled_fasta_headers_true_negative = []
        self.ctxt_relabeled_fasta_seq_true_negative = []
        self.ctxt_relabeled_fasta_headers_predicted_positive = []
        self.ctxt_relabeled_fasta_seq_predicted_positive = []
        self.ctxt_relabeled_fasta_headers_predicted_negative = []
        self.ctxt_relabeled_fasta_seq_predicted_negative = []

        # For SSL
        self.context_labels_ssl = [] 

    def preprocess(self, seq_df, k, step, num_context_kmers, gap_size):
        self.seq_objects, self.split_seq_as_strings, \
            self.split_seq_labels, self.split_seq_ids, \
                self.contexts_as_strings, self.context_labels, \
                    self.context_seq_ids = self.get_all_sequence_contexts(seq_df, k, step, num_context_kmers, gap_size) ### Need to fix where function is coming from
        return(self)

    # Function : get_all_sequence_contexts
    # Description : for each training/test sequence, create and populate a Sequence object and extract all contexts
    # Parameters : 
    # - all_seq_df : a Pandas data frame containing all training/test sequences, along with other relevant information
    # - k : size of the kmers
    # - step : step size of the sliding window
    # - num_context_kmers : number of kmers in the context (left and right)
    # - gap_size : number of kmers to skip in between context kmers
    # Returns : 
    # - all_seq_objects : a list containing all Sequence objects for each of the training/test sequences
    # - all_seq_contexts_as_strings : list of contexts as strings, each in format "left_context target right_context"
    # - all_seq_context_labels : list of labels for each context; labels come from sequence of origin
    # - all_seq_context_seq_ids : list of ids from the sequence each context originates from
    # - all_split_seq_as_strings : list of sequences, where each element in the list is a string of containing the sequence kmers separated by a space
    # - all_split_seq_labels : list of sequence labels
    # - all_split_seq_ids : list of sequence ids
    def get_all_sequence_contexts(self, all_seq_df, k, step, num_context_kmers, gap_size):
        all_seq_objects = []
        all_split_seq_as_strings = []
        all_split_seq_labels = []
        all_split_seq_ids = []
        all_contexts_as_strings = []
        all_context_labels = []
        all_context_seq_ids = []
        seq_counter = 1
        for row in all_seq_df.itertuples():
            seq = contexts.Sequence(row.seq, row.seq_id, encoder.encode_labels(row.y_label), row.CV_group, \
                                    row.chr, row.start, row.end, row.strand)
            seq.split_sequence(k, step)
            seq.convert_split_seq_to_string()      
            all_split_seq_as_strings.append(seq.split_seq_as_strings)
            all_split_seq_labels.append(seq.label)
            all_split_seq_ids.append(seq.id)
            seq.get_seq_contexts(k, step, num_context_kmers, gap_size)
            all_contexts_as_strings += seq.convert_seq_contexts_to_string()
            all_context_labels += seq.get_seq_context_labels()
            all_context_seq_ids += seq.get_seq_context_ids()
            all_seq_objects.append(seq)
            seq_counter += 1
        return(all_seq_objects, all_split_seq_as_strings, all_split_seq_labels, all_split_seq_ids, \
                all_contexts_as_strings, all_context_labels, all_context_seq_ids)
    
    # For baseline model
    def preprocess_sequences(self, all_seq_df, kmer_size_list, step_size_list):
        if (len(self.seq_objects) == 0):
            for row in all_seq_df.itertuples():
                seq = contexts.Sequence(row.seq, row.seq_id, encoder.encode_labels(row.y_label), row.CV_group, \
                                        row.chr, row.start, row.end, row.strand)
                seq.perform_multivariate_split(kmer_size_list, step_size_list)    
                self.split_seq_as_strings += seq.baseline_seq_strings
                self.split_seq_labels += [seq.label] * len(seq.baseline_seq_strings)
                self.split_seq_ids += [seq.id] * len(seq.baseline_seq_strings)
                self.split_seq_cv_groups += [seq.cv_group] * len(seq.baseline_seq_strings)
                self.seq_objects.append(seq)
        else:
            for seq in self.seq_objects:
                seq.perform_multivariate_split(kmer_size_list, step_size_list)    
                self.split_seq_as_strings += seq.baseline_seq_strings
                self.split_seq_labels += [seq.label] * len(seq.baseline_seq_strings)
                self.split_seq_ids += [seq.id] * len(seq.baseline_seq_strings)
                self.split_seq_cv_groups += [seq.cv_group] * len(seq.baseline_seq_strings)
    
    # For contexts model
    def preprocess_contexts(self, all_seq_df, target_size, target_step, kmer_size, kmer_step, num_ctxt):
        if (len(self.seq_objects) == 0):
            for row in all_seq_df.itertuples():
                seq = contexts.Sequence(row.seq, row.seq_id, encoder.encode_labels(row.y_label), row.CV_group, \
                                        row.chr, row.start, row.end, row.strand)
                seq.get_contexts(target_size, target_step, kmer_size, kmer_step, num_ctxt)
                self.contexts_as_strings += seq.context_strings
                self.context_labels += [seq.label] * len(seq.context_objects)
                self.context_seq_ids += [seq.id] * len(seq.context_objects)
                self.context_ids += seq.context_ids
                self.context_cv_groups += [seq.cv_group] * len(seq.context_objects)
                self.seq_objects.append(seq)
        else:
            for seq in self.seq_objects:
                seq.get_contexts(target_size, target_step, kmer_size, kmer_step, num_ctxt)
                self.contexts_as_strings += seq.context_strings
                self.context_labels += [seq.label] * len(seq.context_objects)
                self.context_seq_ids += [seq.id] * len(seq.context_objects)
                self.context_ids += seq.context_ids
                self.context_cv_groups += [seq.cv_group] * len(seq.context_objects)

    # Save the prepUtils object
    def save_object(self, filename):
        filehandler = open(filename, "wb")
        pickle.dump(self, filehandler)
        filehandler.close()
        return()

    # -------------------------------------- NEW CODE ADDED MARCH 1, 2023 ---------------------------------------

    def write_seq_to_fasta(self, predictions_df, threshold):
        if os.path.exists("sequences.test.fa"):
            os.remove("sequences.test.fa")
        if os.path.exists("sequences.test.true_positive.fa"):
            os.remove("sequences.test.true_positive.fa")
        if os.path.exists("sequences.test.true_negative.fa"):
            os.remove("sequences.test.true_negative.fa")
        if os.path.exists("sequences.test.predicted_positive.fa"):
            os.remove("sequences.test.predicted_positive.fa")
        if os.path.exists("sequences.test.predicted_negative.fa"):
            os.remove("sequences.test.predicted_negative.fa")
        # Fasta format
        # >{chr}:{start}-{end}({strand}):{seq_id}:{ctxt_id}
        # sequence
        for seq_obj in self.seq_objects:
            print("Processing " + seq_obj.id + "...")
            header = ">" + seq_obj.chr + ":" + str(seq_obj.start_coord) + "-" + \
                        str(seq_obj.end_coord) + "(" + seq_obj.strand + ")"
            if (header not in self.seq_fasta_headers_all):
                with open("sequences.test.fa", "a") as f:
                    f.write(header + "\n")
                    f.write(seq_obj.seq  + "\n")
                self.seq_fasta_headers_all.append(header)
                self.seq_fasta_seq_all.append(seq_obj.seq)
            seq_prediction = predictions_df[predictions_df.seq_id == seq_obj.id]
            if (float(seq_prediction.true_y) == 1):
                if (header not in self.seq_fasta_headers_true_positive):
                    with open("sequences.test.true_positive.fa", "a") as f:
                        f.write(header + "\n")
                        f.write(seq_obj.seq  + "\n")
                    self.seq_fasta_headers_true_positive.append(header)
                    self.seq_fasta_seq_true_positive.append(seq_obj.seq)
            elif (float(seq_prediction.true_y) == 0):
                if (header not in self.seq_fasta_headers_true_negative):
                    with open("sequences.test.true_negative.fa", "a") as f:
                        f.write(header + "\n")
                        f.write(seq_obj.seq  + "\n")
                    self.seq_fasta_headers_true_negative.append(header)
                    self.seq_fasta_seq_true_negative.append(seq_obj.seq)
            if (float(seq_prediction.predicted_y) >= threshold):
                if (header not in self.seq_fasta_headers_predicted_positive):
                    with open("sequences.test.predicted_positive.fa", "a") as f:
                        f.write(header + "\n")
                        f.write(seq_obj.seq  + "\n")
                    self.seq_fasta_headers_predicted_positive.append(header)
                    self.seq_fasta_seq_predicted_positive.append(seq_obj.seq)
            elif (float(seq_prediction.predicted_y) < threshold):
                if (header not in self.seq_fasta_headers_predicted_negative):
                    with open("sequences.test.predicted_negative.fa", "a") as f:
                        f.write(header + "\n")
                        f.write(seq_obj.seq  + "\n")
                    self.seq_fasta_headers_predicted_negative.append(header)
                    self.seq_fasta_seq_predicted_negative.append(seq_obj.seq)
        return(self)

    def write_ctxt_to_fasta(self, predictions_df, threshold):
        if os.path.exists("contexts.test.fa"):
            os.remove("contexts.test.fa")
        if os.path.exists("contexts.test.true_positive.fa"):
            os.remove("contexts.test.true_positive.fa")
        if os.path.exists("contexts.test.true_negative.fa"):
            os.remove("contexts.test.true_negative.fa")
        if os.path.exists("contexts.test.predicted_positive.fa"):
            os.remove("contexts.test.predicted_positive.fa")
        if os.path.exists("contexts.test.predicted_negative.fa"):
            os.remove("contexts.test.predicted_negative.fa")
        # Fasta format
        # >{chr}:{start}-{end}({strand}):{seq_id}:{ctxt_id}
        # sequence
        for seq_obj in self.seq_objects:
            print("Processing " + seq_obj.id + "...")
            for ctxt_obj in seq_obj.context_objects:
                header = ">" + ctxt_obj.chr + ":" + str(ctxt_obj.start_coord) + "-" + \
                        str(ctxt_obj.end_coord) + "(" + ctxt_obj.strand + ")"
                if (header not in self.ctxt_fasta_headers_all):
                    with open("contexts.test.fa", "a") as f:
                        f.write(header + "\n")
                        f.write(ctxt_obj.original_seq  + "\n")
                    self.ctxt_fasta_headers_all.append(header)
                    self.ctxt_fasta_seq_all.append(ctxt_obj.original_seq)
                ctxt_prediction = predictions_df[predictions_df.seq_id == seq_obj.id]
                ctxt_prediction = ctxt_prediction[ctxt_prediction.ctxt_id == ctxt_obj.id]
                if (float(ctxt_prediction.true_y) == 1):
                    if (header not in self.ctxt_fasta_headers_true_positive):
                        with open("contexts.test.true_positive.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_fasta_headers_true_positive.append(header)
                        self.ctxt_fasta_seq_true_positive.append(ctxt_obj.original_seq)
                elif(float(ctxt_prediction.true_y) == 0):
                    if (header not in self.ctxt_fasta_headers_true_negative):
                        with open("contexts.test.true_negative.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_fasta_headers_true_negative.append(header)
                        self.ctxt_fasta_seq_true_negative.append(ctxt_obj.original_seq)
                if (float(ctxt_prediction.predicted_y) >= threshold):
                    if (header not in self.ctxt_fasta_headers_predicted_positive):
                        with open("contexts.test.predicted_positive.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_fasta_headers_predicted_positive.append(header)
                        self.ctxt_fasta_seq_predicted_positive.append(ctxt_obj.original_seq)
                elif (float(ctxt_prediction.predicted_y) < threshold):
                    if (header not in self.ctxt_fasta_headers_predicted_negative):
                        with open("contexts.test.predicted_negative.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_fasta_headers_predicted_negative.append(header)
                        self.ctxt_fasta_seq_predicted_negative.append(ctxt_obj.original_seq)
        return(self)
    
    def write_baseline_on_ctxt_to_fasta(self, predictions_df, threshold):
        if os.path.exists("baseline_on_contexts.test.fa"):
            os.remove("baseline_on_contexts.test.fa")
        if os.path.exists("baseline_on_contexts.test.true_positive.fa"):
            os.remove("baseline_on_contexts.test.true_positive.fa")
        if os.path.exists("baseline_on_contexts.test.true_negative.fa"):
            os.remove("baseline_on_contexts.test.true_negative.fa")
        if os.path.exists("baseline_on_contexts.test.predicted_positive.fa"):
            os.remove("baseline_on_contexts.test.predicted_positive.fa")
        if os.path.exists("baseline_on_contexts.test.predicted_negative.fa"):
            os.remove("baseline_on_contexts.test.predicted_negative.fa")
        # Fasta format
        # >{chr}:{start}-{end}({strand}):{seq_id}:{ctxt_id}
        # sequence
        for seq_obj in self.seq_objects:
            print("Processing " + seq_obj.id + "...")
            for ctxt_obj in seq_obj.context_objects:
                header = ">" + ctxt_obj.chr + ":" + str(ctxt_obj.start_coord) + "-" + \
                        str(ctxt_obj.end_coord) + "(" + ctxt_obj.strand + ")"
                if (header not in self.baseline_on_ctxt_fasta_headers_all):
                    with open("baseline_on_contexts.test.fa", "a") as f:
                        f.write(header + "\n")
                        f.write(ctxt_obj.original_seq  + "\n")
                    self.baseline_on_ctxt_fasta_headers_all.append(header)
                    self.baseline_on_ctxt_fasta_seq_all.append(ctxt_obj.original_seq)
                ctxt_prediction = predictions_df[predictions_df.seq_id == seq_obj.id]
                ctxt_prediction = ctxt_prediction[ctxt_prediction.ctxt_id == ctxt_obj.id]
                if (float(ctxt_prediction.true_y) == 1):
                    if (header not in self.baseline_on_ctxt_fasta_headers_true_positive):
                        with open("baseline_on_contexts.test.true_positive.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.baseline_on_ctxt_fasta_headers_true_positive.append(header)
                        self.baseline_on_ctxt_fasta_seq_true_positive.append(ctxt_obj.original_seq)
                elif(float(ctxt_prediction.true_y) == 0):
                    if (header not in self.baseline_on_ctxt_fasta_headers_true_negative):
                        with open("baseline_on_contexts.test.true_negative.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.baseline_on_ctxt_fasta_headers_true_negative.append(header)
                        self.baseline_on_ctxt_fasta_seq_true_negative.append(ctxt_obj.original_seq)
                if (float(ctxt_prediction.predicted_y) >= threshold):
                    if (header not in self.baseline_on_ctxt_fasta_headers_predicted_positive):
                        with open("baseline_on_contexts.test.predicted_positive.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.baseline_on_ctxt_fasta_headers_predicted_positive.append(header)
                        self.baseline_on_ctxt_fasta_seq_predicted_positive.append(ctxt_obj.original_seq)
                elif (float(ctxt_prediction.predicted_y) < threshold):
                    if (header not in self.baseline_on_ctxt_fasta_headers_predicted_negative):
                        with open("baseline_on_contexts.test.predicted_negative.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.baseline_on_ctxt_fasta_headers_predicted_negative.append(header)
                        self.baseline_on_ctxt_fasta_seq_predicted_negative.append(ctxt_obj.original_seq)
        return(self)
    
    def write_relabeled_ctxt_to_fasta(self, predictions_df, threshold, iteration):
        # Reinitialize lists for each iteration
        self.ctxt_relabeled_fasta_headers_all = []
        self.ctxt_relabeled_fasta_seq_all = []
        self.ctxt_relabeled_fasta_headers_true_positive = []
        self.ctxt_relabeled_fasta_seq_true_positive = []
        self.ctxt_relabeled_fasta_headers_true_negative = []
        self.ctxt_relabeled_fasta_seq_true_negative = []
        self.ctxt_relabeled_fasta_headers_predicted_positive = []
        self.ctxt_relabeled_fasta_seq_predicted_positive = []
        self.ctxt_relabeled_fasta_headers_predicted_negative = []
        self.ctxt_relabeled_fasta_seq_predicted_negative = []
        if os.path.exists("contexts_relabeled.iteration" + str(iteration) + ".test.fa"):
            os.remove("contexts_relabeled.test.fa")
        if os.path.exists("contexts_relabeled.iteration" + str(iteration) + ".test.true_positive.fa"):
            os.remove("contexts_relabeled.iteration" + str(iteration) + ".test.true_positive.fa")
        if os.path.exists("contexts_relabeled.iteration" + str(iteration) + ".test.true_negative.fa"):
            os.remove("contexts_relabeled.iteration" + str(iteration) + ".test.true_negative.fa")
        if os.path.exists("contexts_relabeled.iteration" + str(iteration) + ".test.predicted_positive.fa"):
            os.remove("contexts_relabeled.iteration" + str(iteration) + ".test.predicted_positive.fa")
        if os.path.exists("contexts_relabeled.iteration" + str(iteration) + ".test.predicted_negative.fa"):
            os.remove("contexts_relabeled.iteration" + str(iteration) + ".test.predicted_negative.fa")
        # Fasta format
        # >{chr}:{start}-{end}({strand}):{seq_id}:{ctxt_id}
        # sequence
        for seq_obj in self.seq_objects:
            print("Processing " + seq_obj.id + "...")
            for ctxt_obj in seq_obj.context_objects:
                header = ">" + ctxt_obj.chr + ":" + str(ctxt_obj.start_coord) + "-" + \
                        str(ctxt_obj.end_coord) + "(" + ctxt_obj.strand + ")"
                if (header not in self.ctxt_relabeled_fasta_headers_all):
                    with open("contexts_relabeled.iteration" + str(iteration) + ".test.fa", "a") as f:
                        f.write(header + "\n")
                        f.write(ctxt_obj.original_seq  + "\n")
                    self.ctxt_relabeled_fasta_headers_all.append(header)
                    self.ctxt_relabeled_fasta_seq_all.append(ctxt_obj.original_seq)
                ctxt_prediction = predictions_df[predictions_df.seq_id == seq_obj.id]
                ctxt_prediction = ctxt_prediction[ctxt_prediction.ctxt_id == ctxt_obj.id]
                if (float(ctxt_prediction.true_y) == 1):
                    if (header not in self.ctxt_relabeled_fasta_headers_true_positive):
                        with open("contexts_relabeled.iteration" + str(iteration) + ".test.true_positive.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_relabeled_fasta_headers_true_positive.append(header)
                        self.ctxt_relabeled_fasta_seq_true_positive.append(ctxt_obj.original_seq)
                elif(float(ctxt_prediction.true_y) == 0):
                    if (header not in self.ctxt_relabeled_fasta_headers_true_negative):
                        with open("contexts_relabeled.iteration" + str(iteration) + ".test.true_negative.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_relabeled_fasta_headers_true_negative.append(header)
                        self.ctxt_relabeled_fasta_seq_true_negative.append(ctxt_obj.original_seq)
                if (float(ctxt_prediction.predicted_y) >= threshold):
                    if (header not in self.ctxt_relabeled_fasta_headers_predicted_positive):
                        with open("contexts_relabeled.iteration" + str(iteration) + ".test.predicted_positive.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_relabeled_fasta_headers_predicted_positive.append(header)
                        self.ctxt_relabeled_fasta_seq_predicted_positive.append(ctxt_obj.original_seq)
                elif (float(ctxt_prediction.predicted_y) < threshold):
                    if (header not in self.ctxt_relabeled_fasta_headers_predicted_negative):
                        with open("contexts_relabeled.iteration" + str(iteration) + ".test.predicted_negative.fa", "a") as f:
                            f.write(header + "\n")
                            f.write(ctxt_obj.original_seq  + "\n")
                        self.ctxt_relabeled_fasta_headers_predicted_negative.append(header)
                        self.ctxt_relabeled_fasta_seq_predicted_negative.append(ctxt_obj.original_seq)
        return(self)
    
    # -------------------------------------- NEW CODE ADDED MAY 1, 2023, FOR SSL ---------------------------------------

    # Count the number of each target in the list of kmers
    def get_target_counts(self, targets_list, targets_unique):
        target_counts = [targets_list.count(target) if target in targets_list else 0 \
                            for target in targets_unique]
        return(target_counts)
    
    # Perform chi-square test using a contingency table to get statistical significance of enrichment
    def perform_chi_square(self, counts_positive, total_positive, counts_negative, total_negative):
        pvalues = []
        for i in range(0, len(counts_positive)):
            observed = [[counts_positive[i], total_positive - counts_positive[i]], \
                            [counts_negative[i], total_negative - counts_negative[i]]]
            chi, p, dof, ex = chi2_contingency(observed)
            pvalues.append(p)
        return(pvalues)

    # Compute the pseudoenrichment zscore
    def compute_enrichment_zscore(self, enrichment_df):                         
        mean_enrichment = np.mean(enrichment_df["enrichment"])
        stdev_enrichment = np.std(enrichment_df["enrichment"])
        enrichment_df["zscore"] = [((x - mean_enrichment) / stdev_enrichment) for x in enrichment_df["enrichment"]]
        return(enrichment_df)

    # Find significant kmers using the provided zscore threshold
    def get_significant_enriched(self, enrichment_df, zscore_threshold):
        signif_enrichment_df = enrichment_df[enrichment_df["zscore"] >= zscore_threshold]
        return(signif_enrichment_df)

    # Get ratio of target counts for positive and negative contexts, using targets only
    def compute_enrichment_targets(self, target_counts_positive, total_positive, \
                                        target_counts_negative, total_negative, target_list_unique, zscore_threshold):
        # Divide counts of each target by the total number of contexts in the respective category
        if (total_positive != 0) and (total_negative != 0):
            target_ratio_positive = [x / total_positive for x in target_counts_positive]
            target_ratio_negative = [x / total_negative for x in target_counts_negative]
            target_enrichment = [target_ratio_positive[i] / target_ratio_negative[i] if target_ratio_negative[i] != 0 else None \
                                    for i in range(0, len(target_ratio_positive))]
            target_pvalues = self.perform_chi_square(target_counts_positive, total_positive, \
                                                    target_counts_negative, total_negative)
            target_enrichment_df = pd.DataFrame({"target": target_list_unique, \
                                                    "count_positive": target_counts_positive, "ratio_positive": target_ratio_positive, \
                                                    "count_negative": target_counts_negative, "ratio_negative": target_ratio_negative, \
                                                    "enrichment": target_enrichment, "pvalue": target_pvalues})
            # Sort by pvalue, then by fold change
            # target_enrichment_df.sort_values(["pvalue", "fold_change"], inplace = True, ascending = [True, False])
            target_enrichment_df.sort_values(["enrichment"], inplace = True, ascending = [False])
            target_enrichment_df.reset_index(inplace = True, drop = True)
            # Perform further processing
            target_enrichment_df = self.compute_enrichment_zscore(target_enrichment_df)
            target_enrichment_df_signif = self.get_significant_enriched(target_enrichment_df, zscore_threshold)
            target_enrichment_df_signif["normalized_enrichment"] = [x / target_enrichment_df_signif["enrichment"].min() \
                                                                        for x in list(target_enrichment_df_signif["enrichment"])]
            target_enrichment_df_signif["proportion"] = [round(x) for x in list(target_enrichment_df_signif["normalized_enrichment"])]
        else:
            target_enrichment_df = pd.DataFrame(0, index = np.arange(len(target_counts_positive)), \
                                                columns = ['target', 'count_positive', 'ratio_positive', 'count_negative', \
                                                                'ratio_negative', 'enrichment', 'pvalue', 'zscore'])
            target_enrichment_df_signif = pd.DataFrame()
        return(target_enrichment_df, target_enrichment_df_signif)

    # Compute kmer enrichment of original data to use for true labeling in SSL
    def compute_kmer_enrichment(self, positive_split_seq_as_strings, negative_split_seq_as_strings, zscore_threshold):
        # Find unique kmers in positive samples
        positive_kmers = [x.split(" ") for x in positive_split_seq_as_strings] # split each space-delimited string into a list of kmers
        positive_kmers = [kmer for sublist in positive_kmers for kmer in sublist] # combine kmers across all sequences to get 1 list of all kmers
        positive_kmers_unique = sorted(list(set(positive_kmers))) # find unique kmers across all sequences
        # Find unique kmers in negative samples
        negative_kmers = [x.split(" ") for x in negative_split_seq_as_strings]
        negative_kmers = [kmer for sublist in negative_kmers for kmer in sublist]
        negative_kmers_unique = sorted(list(set(negative_kmers)))
        # Get counts for each kmer
        all_unique_kmers = sorted(list(set(positive_kmers_unique) & set(negative_kmers_unique))) # get all unique kmers across positive and negative sequences
        positive_kmers_counts = self.get_target_counts(positive_kmers, all_unique_kmers) # count # of each unique kmer in positive sequences
        negative_kmers_counts = self.get_target_counts(negative_kmers, all_unique_kmers) # count # of each unique kmer in negative sequences
        # Compute kmer enrichment
        kmer_enrichment_df, kmer_enrichment_df_signif = self.compute_enrichment_targets(positive_kmers_counts, len(positive_kmers), \
                                                                                        negative_kmers_counts, len(negative_kmers), \
                                                                                        all_unique_kmers, zscore_threshold)
        return(kmer_enrichment_df)
    
    # Update context labels based on kmer enrichments
    def update_labels_by_kmer_enrichment(self, kmer_enrichments):
        # Find most enriched kmer per positive sequence
        for i in range(0, len(self.seq_objects)):
            # Split the space-delimited sequence into a list of kmers
            split_seq = self.split_seq_as_strings[i]
            seq_kmers = split_seq.split(" ")
            max_enrichment = -1
            most_enriched_kmer = ""
            most_enriched_kmer_position = -1
            for j in range(0, len(seq_kmers)):
                kmer = seq_kmers[j]
                # If the current enrichment is greater than the current max, update the max
                if (float(kmer_enrichments[kmer_enrichments.target == kmer].enrichment) >= max_enrichment):
                    max_enrichment = float(kmer_enrichments[kmer_enrichments.target == kmer].enrichment)
                    most_enriched_kmer = kmer
                    most_enriched_kmer_position = j
            # Find all occurrences of the most enriched kmer within the sequence
            indices = [i for i, x in enumerate(seq_kmers) if x == most_enriched_kmer]
            # If the most enriched kmer occurs more than once in the sequence, randomly select an instance to label 1
            if (len(indices) > 1):
                most_enriched_kmer_position = random.sample(indices, 1)[0]
            # Save the most enriched kmer and its position to the Seq object
            self.seq_objects[i].most_enriched_kmer = most_enriched_kmer
            self.seq_objects[i].most_enriched_kmer_position = most_enriched_kmer_position
            # Update the SSL label for the context containing the most enriched kmer as the target
            # Note: ssl_label for all contexts are initialized to 0 when the context object is created
            self.seq_objects[i].context_objects[most_enriched_kmer_position].ssl_label = 1
            seq_ssl_labels = [0] * len(self.seq_objects[i].context_objects)
            seq_ssl_labels[most_enriched_kmer_position] = 1
            self.context_labels_ssl += seq_ssl_labels
        return(self)

    def populate_preprocessor_SSL(self, seq_obj, ctxt_obj):
        self.seq_objects.append(seq_obj)
        self.context_seq_ids.append(seq_obj.id)
        self.context_ids.append(ctxt_obj.id)
        self.context_cv_groups.append(seq_obj.cv_group)
        self.contexts_as_strings.append(ctxt_obj.context_as_string)
        self.context_labels.append(ctxt_obj.seq_label)
        self.context_labels_ssl.append(ctxt_obj.ssl_label)

    def add_contexts_SSL(self, unknown_preprocessor_obj, indices, ssl_label = 1):
        self.seq_objects = self.seq_objects + [unknown_preprocessor_obj.seq_objects[i] for i in indices]
        self.context_seq_ids = self.context_seq_ids + [unknown_preprocessor_obj.context_seq_ids[i] for i in indices]
        self.context_ids = self.context_ids + [unknown_preprocessor_obj.context_ids[i] for i in indices]
        self.context_cv_groups = self.context_cv_groups + [unknown_preprocessor_obj.context_cv_groups[i] for i in indices]
        self.contexts_as_strings = self.contexts_as_strings + [unknown_preprocessor_obj.contexts_as_strings[i] for i in indices]
        self.context_labels = self.context_labels + [unknown_preprocessor_obj.context_labels[i] for i in indices]
        # If the unknown contexts are being added after prediction, their SSL labels must be 1
        # If the unknown contexts are being added at the final stage (no longer predicted 1), their SSL labels must be 0
        self.context_labels_ssl = self.context_labels_ssl + [ssl_label] * len(indices)

    def delete_contexts_SSL(self, indices):
        self.seq_objects = np.delete(self.seq_objects, indices, axis = 0)
        self.context_seq_ids = np.delete(self.context_seq_ids, indices, axis = 0)
        self.context_ids = np.delete(self.context_ids, indices, axis = 0)
        self.context_cv_groups = np.delete(self.context_cv_groups, indices, axis = 0)
        self.contexts_as_strings = np.delete(self.contexts_as_strings, indices, axis = 0)
        self.context_labels = np.delete(self.context_labels, indices, axis = 0)
        self.context_labels_ssl = np.delete(self.context_labels_ssl, indices, axis = 0)

# Function : relabel
# Description : given a probability threshold, relabel contexts based on whether it's predicted probability 
# is above (1) or below (0) the threshold
# Note: only relabel contexts that come from positive sequences (negative sequences should retain label of 0)
# Parameters : 
# - threshold: the probability threshold to use for relabeling; if predicted probability is above the threshold, 
#              the label is relabeled as 1 (0 otherwise)
# - predictions_test_contexts: numpy array containing the predicted probabilities for each context
# Returns :
# - new_labels: the relabeled y's for each context
def relabel(threshold, predictions, old_labels, selective = True):
    # Convert from numpy array to list
    predictions = predictions.flatten().tolist()
    # Initialize new labels
    new_labels = [0] * len(predictions)
    # Update label to 1 if context comes from positive sequence AND has a predicted probability that meets the threshold
    indices_above_threshold = [i for i in range(len(predictions)) if (predictions[i] >= threshold)]
    if (selective == True):
        # Find indices of contexts that come from positive sequences
        positive_indices = [i for i in range(len(old_labels)) if old_labels[i] == 1]
        indices_to_update = list(set(indices_above_threshold) & set(positive_indices))
        indices_to_update.sort()
    else:
        indices_to_update = indices_above_threshold
    updates = [1] * len(indices_to_update)
    for i,j in zip(indices_to_update, range(len(updates))): 
        new_labels[i] = updates[j]
    new_labels = np.array(new_labels)
    return(new_labels)
    



