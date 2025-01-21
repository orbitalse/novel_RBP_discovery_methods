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
    



