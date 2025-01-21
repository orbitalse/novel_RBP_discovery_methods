# Shaimae Elhajjajy
# July 14, 2022
# Functions to encode training and testing data (X and y)

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Import libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------- CLASSES -------------------------------------------------------------

class Tokenizer:
    # Initialize Tokenizer object
    def __init__(self):
        self.tokenizer = keras.preprocessing.text.Tokenizer()
        self.vocab_size = None
        self.word_index = None
        self.word_counts = None
        self.word_docs = None
        self.document_count = None
        self.fitted = False
    
    # Function : fit
    # Description : initialize and fit the tokenizer
    # Parameters : 
    # - training_strings: sequences/contexts for training data as strings; used to fit the tokenizer
    # Returns :
    # - tokenizer: the tokenizer object, fit on training sequences/contexts
    def fit(self, training_strings):
        if (self.fitted == False):
            # Create and fit the tokenizer
            self.tokenizer.fit_on_texts(training_strings)

            self.tokenizer = self.tokenizer
            self.vocab_size = len(self.tokenizer.word_counts) # num kmers detected in all contexts
            self.word_index = self.tokenizer.word_index
            self.word_counts = self.tokenizer.word_counts
            self.word_docs = self.tokenizer.word_docs
            self.document_count = self.tokenizer.document_count

            self.fitted = True

        return(self)

    #  Add description
    def refit(self, training_strings):
        self.fitted = False
        self.fit(self, training_strings)
        return(self)
    
    # Function : encode_matrix
    # Description : encode the kmers in each sequence/context
    # Parameters :
    # - self: the Tokenizer object, fit on training sequences/contexts
    # - seq_list: sequences/contexts for training data as strings
    # - mode: method to encode as matrix (binary, count, freq, tfidf)
    # Returns : 
    # - X: encoded matrix for sequence data
    def encode_matrix(self, seq_list, mode):
        # Encode sequences and contexts according to specified mode
        X = self.tokenizer.texts_to_matrix(seq_list, mode = mode) # num training sequences/contexts x vocab size

        return(X)

    # Function : FindMaxLength
    # Description : find the longest list within a nested list
    # Note: from https://www.geeksforgeeks.org/python-find-maximum-length-sub-list-in-a-nested-list/
    # Parameters :
    # - lst : nested list
    # Returns : 
    # maxLength : the length of the longest list in the nested list
    def FindMaxLength(self, lst):
        maxList = max(lst, key = lambda i: len(i))
        maxLength = len(maxList)
        return(maxLength)

    # Function : pad_contexts_11_old
    # Description : pre-pad left contexts and post-pad right contexts
    # Parameters :
    # - X : nested list of contexts that have been integer encoded
    # Returns : 
    # - X_padded : np array of contexts that have been appropriately padded
    def pad_contexts_11_old(self, X, num_ctxt_per_seq):
        # Append padded sequences to the main list
        X_padded = []
        # Calculate parameters
        total_num_ctxts = len(X) # total number of contexts from all sequences
        num_seq = int(total_num_ctxts / num_ctxt_per_seq) # total number of sequences in dataset (training or test)
        seq_boundaries = range(0, total_num_ctxts + num_ctxt_per_seq, num_ctxt_per_seq) # set boundaries to index contexts from each sequence
        max_length = self.FindMaxLength(X) # Find max length of a full context (e.g., # elements in left context, target, right context)
        midpoint = num_ctxt_per_seq / 2 # Set a threshold to determine whether a given context is from the beginning or end of a sequence
        # Loop through contexts to pad as necessary
        for i in range(0, num_seq):
            # Extract contexts for each sequence
            subset = X[seq_boundaries[i]:seq_boundaries[i + 1]]
            for j in range(0, len(subset)):
                # If the context comes from the beginning of the sequence, pre-pad
                if (len(subset[j]) < max_length) and (j < midpoint):
                    padded_seq = pad_sequences([subset[j]], maxlen = max_length, padding = "pre")
                    X_padded.append(padded_seq.tolist()[0])
                # If the context comes from the end of the sequence, post-pad
                elif (len(subset[j]) < max_length) and (j > midpoint):
                    padded_seq = pad_sequences([subset[j]], maxlen = max_length, padding = "post")
                    X_padded.append(padded_seq.tolist()[0])
                # If the context comes from the beginning of the sequence and has max number of elements, don't pad
                elif (len(subset[j]) == max_length):
                    X_padded.append(subset[j])             
        # Convert to numpy array
        X_padded = np.array(X_padded)
        return(X_padded, max_length)
    
    # New version of pad_contexts_11 written May 2, 2023
    # Note: this version is simpler and more efficient. 
    def pad_contexts_11(self, X, contexts_as_strings):
        # Append padded sequences to the main list
        X_padded = []
        # Calculate parameters
        max_length = self.FindMaxLength(X) # Find max length of a full context (e.g., # elements in left context, target, right context)
        # Loop through contexts to pad as necessary
        for i in range(0, len(X)):
            context_tokenization = X[i]
            context_string = contexts_as_strings[i]
            # If the context has the maximum number of elements, don't need to pad
            if (len(context_tokenization) == max_length):
                X_padded.append(context_tokenization)
            # If the context does not have the maximum number of elements, need to pad at beginning or end
            else:
                padded_seq = [0] * max_length
                # Calculate the context insertion index
                context_split_string = context_string.split(" ")
                # Pad beginning
                if (context_split_string[-1] != ""):
                    empty_indices = [i for i, x in enumerate(context_split_string) if x == ""]
                    insertion_start_index = empty_indices[-1] + 1
                    padded_seq[insertion_start_index : len(padded_seq)] = context_tokenization
                # Pad end
                elif (context_split_string[-1] == ""):
                    empty_indices = [i for i, x in enumerate(context_split_string) if x == ""]
                    insertion_end_index = empty_indices[0]
                    padded_seq[0 : insertion_end_index] = context_tokenization
                X_padded.append(padded_seq)      
        # Convert to numpy array
        X_padded = np.array(X_padded)
        return(X_padded, max_length)

    def pad_contexts_97(self, X, contexts_as_strings, num_ctxt, num_ctxt_per_seq):
        # Append padded sequences to the main list
        X_padded = []
        # Calculate parameters
        total_num_ctxts = len(X) # total number of contexts from all sequences
        num_seq = int(total_num_ctxts / num_ctxt_per_seq) # total number of sequences in dataset (training or test)
        seq_boundaries = range(0, total_num_ctxts + num_ctxt_per_seq, num_ctxt_per_seq)
        # Loop through contexts to pad as necessary
        for i in range(0, num_seq):
            # Extract contexts for each sequence
            subset = X[seq_boundaries[i]:seq_boundaries[i + 1]]
            ctxts = contexts_as_strings[seq_boundaries[i]:seq_boundaries[i + 1]]
            # Pre- or post-pad contexts based on their position within the sequence
            for j in range(0, len(subset)):
                padded_seq = [0] * len(subset)
                target_insertion_index = j
                # Calculate the context insertion index
                ctxt = ctxts[j].split(" ")
                num_empty = ctxt.count("")
                if (ctxt[-1] != ""):
                    insertion_index = target_insertion_index - (num_ctxt - num_empty)
                elif (ctxt[-1] == ""):
                    insertion_index = target_insertion_index - num_ctxt
                padded_seq[insertion_index : (len(subset[j]) + insertion_index)] = subset[j]
                X_padded.append(padded_seq)
        # Convert to numpy array
        X_padded = np.array(X_padded)
        max_length = num_ctxt_per_seq
        return(X_padded, max_length)

    # Function : encode_integers_baseline
    # Description : encode sequences as a list of integers
    # Parameters : 
    # - self: the Tokenizer object, fit on training sequences
    # - seq_list: sequences for training data as strings
    # Returns : 
    # - X: encoded matrix for sequence data
    def encode_integers_baseline(self, seq_list):
        # Encode sequences as a list of integers
        X = self.tokenizer.texts_to_sequences(seq_list)
        max_length = self.FindMaxLength(X) # Find max length of a full sequence
        X_padded = pad_sequences(X, maxlen = max_length, padding = "post") # In baeline, no padding will take place since all sequences are the same length
        return(X_padded, max_length)
    
    # Function : encode_integers_contexts
    # Description : encode contexts as a list of integers
    # Parameters : 
    # - self: the Tokenizer object, fit on training contexts
    # - seq_list: contexts for training data as strings
    # Returns : 
    # - X: encoded matrix for sequence data
    def encode_integers_contexts(self, contexts_as_strings, num_ctxt, num_ctxt_per_seq):
        # Encode contexts as a list of integers
        X = self.tokenizer.texts_to_sequences(contexts_as_strings)
        X_padded, max_length = self.pad_contexts_11(X, contexts_as_strings)
        # X_padded, max_length = self.pad_contexts_97(X, contexts_as_strings, num_ctxt, num_ctxt_per_seq)
        return(X_padded, max_length)

# ------------------------------------------------------------- FUNCTIONS -------------------------------------------------------------

# Function : encode_labels
# Description : Encode the "bound" and "unbound" labels as 1 or 0, respectively
# Parameters : 
# - label : a string indicating whether the sequence is bound or unbound
# Returns : 
# - label_encoded : an integer indicating whether the sequence is bound (1) or unbound (0)
def encode_labels(label):
    label_dict = {
        "unbound" : 0,
        "bound" : 1,
    }
    label_encoded = label_dict[label]
    return(label_encoded)


    
