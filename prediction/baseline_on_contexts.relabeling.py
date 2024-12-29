# Shaimae Elhajjajy
# July 29, 2022
# Run baseline model (trained on sequences split on kmers) on the contexts
# Note: In this version, baseline model is with BoW, contexts model is with embedding
# This is a solution to difference in embedding size between sequences and contexts
# Refit the same contexts model on relabeled sequences.

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Load libraries
from datetime import datetime
import json
import numpy as np
import pandas as pd
import random
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import sys
from tensorflow import keras

# Load custom classes and functions
import classifier
import contexts
import cv
import readData # function for reading in sequences and models from file
import encoder # encoding labels and tokenizer
import prepUtils

if len(sys.argv) != 12:
    sys.exit("Enter:\n (1) file containing training sequences\n (2) the file containing test sequences\n \
(3) list of baseline kmer sizes (comma-separated)\n \
(4) list of baseline step sizes (comma-separated)\n (5) context target size\n (6) context target step\n \
(7) context kmer size\n (8) context kmer step\n (9) number of context kmers on each side of target\n \
(10) threshold for relabeling\n (11) number of relabeling iterations.")

training_seq_file = sys.argv[1]
test_seq_file = sys.argv[2]
baseline_kmer_size_list = [int(x) for x in sys.argv[3].split(",")]
baseline_step_size_list = [int(x) for x in sys.argv[4].split(",")]
context_target_size = int(sys.argv[5])
context_target_step = int(sys.argv[6])
context_kmer_size = int(sys.argv[7])
context_kmer_step = int(sys.argv[8])
num_ctxt = int(sys.argv[9])
threshold = float(sys.argv[10])
num_iterations = int(sys.argv[11])

# ----------------------------------------------------- FUNCTIONS -----------------------------------------------------

# Function : format_split_seq_output
# Description : create a data frame summarizing predictions, for model trained on split sequence kmers
# Parameters : 
# - test_split_seq_as_strings: list of all test sequences, each as a string of kmers
# - test_split_seq_ids: list of sequence IDs
# - y_test_split_seq: np array of the true labels for test sequences
# - predictions_test_split_seq: np array of predicted probabilities for text contexts
# Returns :
# - output_df: Pandas dataframe summarizing predictions for sequences, with sequence ID, kmers in split sequence, and true/predicted label
def format_split_seq_output(test_split_seq_as_strings, test_split_seq_ids, y_test_split_seq, predictions_test_split_seq):
    y_test_list = y_test_split_seq.tolist() # Convert true labels from np array to list
    test_predictions_list = [prediction[0] for prediction in predictions_test_split_seq.tolist()] # Convert predictions from np array to list
    # Create summary dataframe
    output_df = pd.DataFrame({"seq_id" : test_split_seq_ids, "split_seq" : test_split_seq_as_strings, \
                                 "true_y" : y_test_list, "predicted_y" : test_predictions_list}) 
    return(output_df)

# Function : format_contexts_output
# Description : create data frame summarizing predictions, for model trained on kmer contexts
# Parameters : 
# - test_contexts_as_strings: list of all test contexts, each as a string
# - test_contexts_seq_ids: list of sequence IDs, denoting which sequence a context originates from
# - y_test_contexts: np array of true labels for test contexts
# - predictions_test_contexts: np array of predicted probabilities for text contexts
# - num_context_kmers: number of kmers in context
# Returns :
# - output_df: Pandas dataframe summarizing predictions for contexts, with sequence of origin, target kmer, left/right contexts, and true/predicted label
def format_contexts_output(test_contexts_as_strings, test_contexts_seq_ids, test_contexts_ids, \
                            y_test_contexts, predictions_test_contexts, num_context_kmers):
    # Reformat list of all contexts into nested list, in which each individual list contains a separate context
    context_list = [context.split(" ") for context in test_contexts_as_strings]
    # Get list of all left contexts
    left_contexts = [" ".join(context[0:num_context_kmers]) for context in context_list]
    # Get list of all targets
    targets = [context[num_context_kmers] for context in context_list]
    # Get list of all right contexts
    right_contexts = [" ".join(context[(num_context_kmers + 1):len(context)]) for context in context_list]
    y_test_list = y_test_contexts.tolist() # Convert true labels from np array to list
    test_predictions_list = [prediction[0] for prediction in predictions_test_contexts.tolist()] # Convert predictions from np array to list
    # Create summary dataframe
    output_df = pd.DataFrame({"seq_id" : test_contexts_seq_ids, "ctxt_id" : test_contexts_ids, \
                                "left_context" : left_contexts, "targets" : targets, "right_context" : right_contexts, \
                                 "true_y" : y_test_list, "predicted_y" : test_predictions_list}) 
    return(output_df)

# Function : get_class_distribution
# Description : Find the distribution of positive and negative samples (to get the PR curve baseline)
# Parameters :
# - y_train : The training labels
# - y_test : The test labels
# Results :
# - class_distribution : A dataframe with the number of positives and negatives
def get_class_distribution(y_train, y_test):
    num_positive = len(np.where(y_train == 1)[0]) + len(np.where(y_test == 1)[0])
    num_negative = len(np.where(y_train == 0)[0]) + len(np.where(y_test == 0)[0])
    class_distribution = pd.DataFrame({"num_positive": [num_positive], "num_negative": [num_negative]})
    return(class_distribution)

# Function : get_class_distribution
# Description : Find the distribution of positive and negative samples (to get the PR curve baseline)
# Parameters :
# - y_train : The training labels
# - y_test : The test labels
# Results :
# - class_distribution : A dataframe with the number of positives and negatives
def get_class_distribution_test(y_test):
    num_positive = len(np.where(y_test == 1)[0])
    num_negative = len(np.where(y_test == 0)[0])
    class_distribution = pd.DataFrame({"num_positive": [num_positive], "num_negative": [num_negative]})
    return(class_distribution)

# Function : FindMaxLength
# Description : find the longest list within a nested list
# Note: from https://www.geeksforgeeks.org/python-find-maximum-length-sub-list-in-a-nested-list/
# Parameters :
# - lst : nested list
# Returns : 
# maxLength : the length of the longest list in the nested list
def FindMaxLength(lst):
    maxList = max(lst, key = lambda i: len(i))
    maxLength = len(maxList)
    return(maxLength)

# ----------------------------------------------------- MAIN -----------------------------------------------------

if __name__ == "__main__":

#    # Read in sequences
#    all_seq_df = readData.load_seq(all_seq_file)
#    # Add a sequence identifier to the dataframe
#    all_seq_df["seq_id"] = ["seq_" + str(i + 1) for i in range(0, all_seq_df.shape[0])]

    # Separate into training and test sets
#    all_cv_groups = list(all_seq_df.CV_group.unique())
#    chromCV = cv.ChromCV(all_cv_groups)
#    training_seq_df, test_seq_df = chromCV.split_train_test_CV(all_seq_df)

    training_seq_df = readData.load_seq(training_seq_file)
    training_seq_df["seq_id"] = ["seq_" + str(i + 1) for i in range(0, training_seq_df.shape[0])]
    test_seq_df = readData.load_seq(test_seq_file)
    test_seq_df["seq_id"] = ["seq_" + str(i + 1) for i in range(training_seq_df.shape[0], training_seq_df.shape[0] + test_seq_df.shape[0])]

    training_cv_groups = sorted(list(training_seq_df.CV_group.unique()))
    test_cv_group = sorted(list(test_seq_df.CV_group.unique()))
    all_cv_groups = training_cv_groups + test_cv_group
    chromCV = cv.ChromCV(all_cv_groups)
    chromCV.training_cv_groups = training_cv_groups
    chromCV.test_cv_group = test_cv_group
    chromCV.training_seq_df = training_seq_df
    chromCV.test_seq_df = test_seq_df

    # Split sequences into kmers
    training_preprocessor = prepUtils.Preprocessor()
    training_preprocessor.preprocess_sequences(training_seq_df, baseline_kmer_size_list, baseline_step_size_list)
    test_preprocessor = prepUtils.Preprocessor()
    test_preprocessor.preprocess_sequences(test_seq_df, baseline_kmer_size_list, baseline_step_size_list)

    # ---------------------------------------- READ IN HYPERPARAMETERS FROM TUNING ----------------------------------------

    baseline_hyperparameters_df = pd.read_csv("baseline_hyperparameters.tsv", sep = "\t", header = None)
    baseline_hyperparameters_df.columns = ["hyperparam", "value"]

    baseline_model_type = baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "baseline_model_type"].value.iloc[0]
    baseline_embedding_type = baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "baseline_embedding_type"].value.iloc[0]

    baseline_hidden_shape = int(baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "hidden_shape"].value.iloc[0])
    baseline_embedding_dim = int(baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "embedding_dim"].value.iloc[0])
    baseline_num_filters = int(baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "num_filters"].value.iloc[0])
    baseline_kernel_length = int(baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "kernel_length"].value.iloc[0])
    baseline_epochs = int(baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "epochs"].value.iloc[0])
    baseline_batch_size = int(baseline_hyperparameters_df[baseline_hyperparameters_df.hyperparam == "batch_size"].value.iloc[0])

    # Set up summary output file
    summary_output_file = "run_summary.tsv"
    with open(summary_output_file, 'a') as f:
        print("----- Baseline model hyperparameters -----", file = f)
        print("\tBaseline model type:" + "\t" + str(baseline_model_type), file = f)
        print("\tBaseline embedding type:" + "\t" + str(baseline_embedding_type), file = f)
        print("\tHidden shape:" + "\t" + str(baseline_hidden_shape), file = f)
        print("\tEmbedding dimension:" + "\t" + str(baseline_embedding_dim), file = f)
        print("\tNumber of filters: " + str(baseline_num_filters), file = f)
        print("\tKernel length:" + "\t" + str(baseline_kernel_length), file = f)
        print("\tNumber of epochs:" + "\t" + str(baseline_epochs), file = f)
        print("\tBatch size: " + str(baseline_batch_size), file = f)
        print("\n", file = f)

    # --------------------------------------------- SET UP BASELINE MODEL ---------------------------------------------

    print("Training baseline model...")

    # Initialize baseline model
    baseline_model = classifier.Classifier()

    # Fit the tokenizer
    baseline_tokenizer = baseline_model.tokenize(training_preprocessor.split_seq_as_strings)
    baseline_vocab_size = baseline_tokenizer.vocab_size + 1

    # Define encoding method (binary, count, freq, tfidf)
    baseline_mode = "count"
    # Encode X as a count matrix (Bag-of-Words)
    X_train_split_seq = baseline_model.encode_matrix(training_preprocessor.split_seq_as_strings, baseline_mode)
    X_test_split_seq = baseline_model.encode_matrix(test_preprocessor.split_seq_as_strings, baseline_mode)
    baseline_max_length = max(FindMaxLength(X_train_split_seq), FindMaxLength(X_test_split_seq))

    # Format y
    y_train_split_seq = np.asarray(training_preprocessor.split_seq_labels) # length is total num sequences in training set
    y_test_split_seq = np.asarray(test_preprocessor.split_seq_labels) # length is total num sequences in test set

    # Set up summary output file
    with open(summary_output_file, 'a') as f:
        print("----- Baseline model parameters -----", file = f)
        print("\tKmer sizes:" + "\t" + str(baseline_kmer_size_list), file = f)
        print("\tStep sizes:" + "\t" + str(baseline_step_size_list), file = f)
        print("\tVocab size: " + str(baseline_model.tokenizer.vocab_size), file = f)
        print("\n", file = f)

    # --------------------------------------------- RUN CV on BASELINE MODEL ---------------------------------------------

    # Set input and output shape of data going into and out of the model
    baseline_model.in_shape = baseline_vocab_size
    baseline_model.hidden_shape = baseline_hidden_shape
    baseline_model.out_shape = 1
    baseline_model.embedding_dim = baseline_embedding_dim
    baseline_model.num_filters = baseline_num_filters
    baseline_model.kernel_length = baseline_kernel_length
    baseline_model.kernel_stride = 1
    baseline_model.pool_size = 2
    baseline_model.epochs = baseline_epochs
    baseline_model.batch_size = baseline_batch_size

    baseline_hyperparams = {
        "hidden_shape": [baseline_model.hidden_shape],
        "loss": ["binary_crossentropy"],
        "optimizer": ["adam"],
        "epochs": [baseline_model.epochs],
        "batch_size": [baseline_model.batch_size]
    }

    # Save the baseline class distribution
    baseline_class_distribution = get_class_distribution_test(y_test_split_seq)
    baseline_class_distribution.to_csv("baseline.class_distributions.tsv", sep = "\t", index = False)

    ## Run baseline model
    # Get indices for chromCV iterations for baseline sequences
    chromCV_iterable_baseline = chromCV.format_CV_iterable_baseline(training_preprocessor)

    # Perform cross-validation and hyperparameter tuning
    baseline_search = cv.run_CV(baseline_model_type, baseline_model, baseline_vocab_size, baseline_max_length, \
                                baseline_hyperparams, chromCV_iterable_baseline, \
                                cv.custom_scoring_function_baseline, X_train_split_seq, y_train_split_seq)

    baseline_CV_accuracy_df, \
            baseline_CV_auroc_df, \
                baseline_CV_aupr_df = cv.assess_CV(baseline_search, chromCV.training_cv_groups)
    baseline_CV_accuracy_df.to_csv("baseline_performance.CV.accuracy.tsv", sep = "\t", index = False)
    baseline_CV_auroc_df.to_csv("baseline_performance.CV.auroc.tsv", sep = "\t", index = False)
    baseline_CV_aupr_df.to_csv("baseline_performance.CV.aupr.tsv", sep = "\t", index = False)

    # --------------------------------------------- RUN FINAL BASELINE MODEL ---------------------------------------------

    # Train final baseline model
    # dense model with BoW encoding
    if (baseline_model_type == "dense"):
        baseline_model.build_BoW()
    # dense model with embedding layer
    elif (baseline_model_type == "dense_embed"):
        baseline_model.build_w_Embedding(baseline_vocab_size, baseline_max_length)
    # cnn with embedding layer
    elif (baseline_model_type == "cnn"):
        baseline_model.build_w_CNN(baseline_vocab_size, baseline_max_length)
    # cnn + lstm with embedding layer
    elif (baseline_model_type == "cnn_lstm"):
        baseline_model.build_w_CNN_LSTM(baseline_vocab_size, baseline_max_length)
    # cnn + bilstm with embedding layer
    elif (baseline_model_type == "cnn_bilstm"):
        baseline_model.build_w_CNN_biLSTM(baseline_vocab_size, baseline_max_length)
    # cnn + gru with embedding layer
    elif (baseline_model_type == "cnn_gru"):
        baseline_model.build_w_CNN_GRU(baseline_vocab_size, baseline_max_length)

    baseline_model.fit(X_train_split_seq, y_train_split_seq)
    predictions_train_split_seq = baseline_model.predict(X_train_split_seq)
    predictions_test_split_seq = baseline_model.predict(X_test_split_seq)

    baseline_train_loss, baseline_train_acc, \
        baseline_train_auroc, baseline_train_aupr = baseline_model.evaluate_plus(X_train_split_seq, \
                                                                                y_train_split_seq, \
                                                                                predictions_train_split_seq, \
                                                                                "baseline_predictions.training")
    baseline_test_loss, baseline_test_acc, \
        baseline_test_auroc, baseline_test_aupr = baseline_model.evaluate_plus(X_test_split_seq, \
                                                                                y_test_split_seq, \
                                                                                predictions_test_split_seq, \
                                                                                "baseline_predictions.test")

    # Format output for display of results
    baseline_train_output_df = format_split_seq_output(training_preprocessor.split_seq_as_strings, \
                                                        training_preprocessor.split_seq_ids, \
                                                        y_train_split_seq, predictions_train_split_seq)
    baseline_test_output_df = format_split_seq_output(test_preprocessor.split_seq_as_strings, \
                                                        test_preprocessor.split_seq_ids, \
                                                        y_test_split_seq, predictions_test_split_seq)

    # Save results to file
    baseline_train_output_filename = "baseline_predictions.training.tsv"
    baseline_train_output_df.to_csv(baseline_train_output_filename, sep = "\t", header = False, index = False)
    baseline_test_output_filename = "baseline_predictions.test.tsv"
    baseline_test_output_df.to_csv(baseline_test_output_filename, sep = "\t", header = False, index = False)

    # Write sequences to a fasta file
    #test_preprocessor.write_seq_to_fasta(baseline_test_output_df, threshold)

    # Print summary of baseline results
    with open(summary_output_file, 'a') as f:
        print("----- Baseline model ----- ", file = f)
        print("\tTraining Accuracy:" + "\t" + str(baseline_train_acc), file = f)
        print("\tTraining AUROC:" + "\t" + str(baseline_train_auroc), file = f)
        print("\tTraining AUPR:" + "\t" + str(baseline_train_aupr), file = f)
        print("", file = f)
        print("\tTest Accuracy:" + "\t" + str(baseline_test_acc), file = f)
        print("\tTest AUROC:" + "\t" + str(baseline_test_auroc), file = f)
        print("\tTest AUPR:" + "\t" + str(baseline_test_aupr), file = f)
        print("", file = f)
        print("\tAvg CV Train AUROC:" + "\t" + str(baseline_CV_auroc_df.loc["avg"]["train"]) + " +/- " + \
                                                    str(baseline_CV_auroc_df.loc["stdev"]["train"]), file = f)
        print("\tAvg CV Test AUROC:" + "\t" + str(baseline_CV_auroc_df.loc["avg"]["test"]) + " +/- " + \
                                                    str(baseline_CV_auroc_df.loc["stdev"]["test"]), file = f)

    # # Save model
    # baseline_model_filename = "baseline_model.keras"
    # baseline_model.save_model(baseline_model_filename)

    # Save objects
    baseline_model.save_object("baseline_model.obj")

    # ---------------------------------------- READ IN HYPERPARAMETERS FROM TUNING ----------------------------------------

    contexts_hyperparameters_df = pd.read_csv("contexts_hyperparameters.tsv", sep = "\t", header = None)
    contexts_hyperparameters_df.columns = ["hyperparam", "value"]

    contexts_model_type = contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "contexts_model_type"].value.iloc[0]
    contexts_embedding_type = contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "contexts_emmbedding_type"].value.iloc[0]

    contexts_hidden_shape = int(contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "hidden_shape"].value.iloc[0])
    contexts_embedding_dim = int(contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "embedding_dim"].value.iloc[0])
    contexts_num_filters = int(contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "num_filters"].value.iloc[0])
    contexts_kernel_length = int(contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "kernel_length"].value.iloc[0])
    contexts_epochs = int(contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "epochs"].value.iloc[0])
    contexts_batch_size = int(contexts_hyperparameters_df[contexts_hyperparameters_df.hyperparam == "batch_size"].value.iloc[0])

    # Set up summary output file
    summary_output_file = "run_summary.tsv"
    with open(summary_output_file, 'a') as f:
        print("----- Contexts model hyperparameters -----", file = f)
        print("\tContexts model type:" + "\t" + str(contexts_model_type), file = f)
        print("\tContexts embedding type:" + "\t" + str(contexts_embedding_type), file = f)
        print("\tHidden shape:" + "\t" + str(contexts_hidden_shape), file = f)
        print("\tEmbedding dimension:" + "\t" + str(contexts_embedding_dim), file = f)
        print("\tNumber of filters: " + str(contexts_num_filters), file = f)
        print("\tKernel length:" + "\t" + str(contexts_kernel_length), file = f)
        print("\tNumber of epochs:" + "\t" + str(contexts_epochs), file = f)
        print("\tBatch size: " + str(contexts_batch_size), file = f)
        print("\n", file = f)

    # ---------------------------------------------- SET UP CONTEXTS MODEL ----------------------------------------------

    print("Training contexts model...")

    # Split sequences into contexts
    training_preprocessor.preprocess_contexts(training_seq_df, context_target_size, context_target_step, context_kmer_size, context_kmer_step, num_ctxt)
    test_preprocessor.preprocess_contexts(test_seq_df, context_target_size, context_target_step, context_kmer_size, context_kmer_step, num_ctxt)

    # Initialize contexts model
    contexts_model = classifier.Classifier()

    # Fit the tokenizer
    contexts_tokenizer = contexts_model.tokenize(training_preprocessor.contexts_as_strings)
    contexts_vocab_size = contexts_tokenizer.vocab_size + 1

    # Format X
    if (contexts_embedding_type == "BoW"):
        # Define encoding method (binary, count, freq, tfidf)
        contexts_mode = "count"
        # Encode X as a count matrix (Bag-of-Words)
        X_train_contexts = contexts_model.encode_matrix(training_preprocessor.contexts_as_strings, contexts_mode)
        X_test_contexts = contexts_model.encode_matrix(test_preprocessor.contexts_as_strings, contexts_mode)
    elif (contexts_embedding_type == "embedding"):
        # Encode X as series of integers
        num_ctxt_per_seq = len(training_preprocessor.seq_objects[0].context_objects)
        X_train_contexts, contexts_train_max_length = contexts_model.encode_integers_contexts(training_preprocessor.contexts_as_strings, \
                                                                                                num_ctxt, num_ctxt_per_seq)
        X_test_contexts, contexts_test_max_length = contexts_model.encode_integers_contexts(test_preprocessor.contexts_as_strings, \
                                                                                                num_ctxt, num_ctxt_per_seq)
        contexts_max_length = max(contexts_train_max_length, contexts_test_max_length)

    # Format y
    y_train_contexts = np.asarray(training_preprocessor.context_labels) # length is total num sequences in training set * num contexts per sequence
    y_test_contexts = np.asarray(test_preprocessor.context_labels) # length is total num sequences in test set * num contexts per sequence

    # Write to summary output file
    with open(summary_output_file, 'a') as f:
        print("----- Contexts model parameters -----", file = f)
        print("\tTarget size:" + "\t" + str(context_target_size), file = f)
        print("\tTarget step:" + "\t" + str(context_target_step), file = f)
        print("\tKmer size:" + "\t" + str(context_kmer_size), file = f)
        print("\tKmer step:" + "\t" + str(context_kmer_step), file = f)
        print("\tNumber of contexts:" + "\t" + str(num_ctxt), file = f)
        print("\tVocab size: " + str(contexts_model.tokenizer.vocab_size), file = f)
        print("", file = f)

    # --------------------------------------------- RUN CV ON MODEL FOR CONTEXTS ---------------------------------------------

    # Set input and output shape of data going into and out of the model
    contexts_model.in_shape = X_train_contexts.shape[1]
    contexts_model.hidden_shape = contexts_hidden_shape
    contexts_model.out_shape = 1
    contexts_model.embedding_dim = contexts_embedding_dim
    contexts_model.num_filters = contexts_num_filters
    contexts_model.kernel_length = contexts_kernel_length
    contexts_model.kernel_stride = 1
    contexts_model.pool_size = 2
    contexts_model.epochs = contexts_epochs
    contexts_model.batch_size = contexts_batch_size

    contexts_hyperparams = {
        "hidden_shape": [contexts_model.hidden_shape],
        "loss": ["binary_crossentropy"],
        "optimizer": ["adam"],
        "epochs": [contexts_model.epochs],
        "batch_size": [contexts_model.batch_size],
    }

    # Save the contexts class distribution
    contexts_class_distribution = get_class_distribution_test(y_test_contexts)
    contexts_class_distribution.to_csv("contexts.class_distributions.tsv", sep = "\t", index = False)

    ## Run contexts model
    # Get indices for chromCV iterations for contexts
    chromCV_iterable_contexts = chromCV.format_CV_iterable_contexts(training_preprocessor)

    # Perform cross-validation and hyperparameter tuning
    contexts_search = cv.run_CV(contexts_model_type, contexts_model, contexts_vocab_size, contexts_max_length, \
                                contexts_hyperparams, chromCV_iterable_contexts, \
                                cv.custom_scoring_function_contexts, X_train_contexts, y_train_contexts)

    contexts_CV_accuracy_df, \
            contexts_CV_auroc_df, \
                contexts_CV_aupr_df = cv.assess_CV(contexts_search, chromCV.training_cv_groups)
    contexts_CV_accuracy_df.to_csv("contexts_performance.CV.accuracy.tsv", sep = "\t", index = False)
    contexts_CV_auroc_df.to_csv("contexts_performance.CV.auroc.tsv", sep = "\t", index = False)
    contexts_CV_aupr_df.to_csv("contexts_performance.CV.aupr.tsv", sep = "\t", index = False)

    # --------------------------------------------- RUN FINAL MODEL FOR CONTEXTS ---------------------------------------------

    # dense model with BoW encoding
    if (contexts_model_type == "dense"):
        contexts_model.build()
    # dense model with embedding layer
    elif (contexts_model_type == "dense_embed"):
        contexts_model.build_w_Embedding(contexts_vocab_size, contexts_max_length)
    # cnn with embedding layer
    elif (contexts_model_type == "cnn"):
        contexts_model.build_w_CNN(contexts_vocab_size, contexts_max_length)
    # cnn + lstm with embedding layer
    elif (contexts_model_type == "cnn_lstm"):
        contexts_model.build_w_CNN_LSTM(contexts_vocab_size, contexts_max_length)
    # cnn + bilstm with embedding layer
    elif (contexts_model_type == "cnn_bilstm"):
        contexts_model.build_w_CNN_biLSTM(contexts_vocab_size, contexts_max_length)
    # cnn + gru with embedding layer
    elif (contexts_model_type == "cnn_gru"):
        contexts_model.build_w_CNN_GRU(contexts_vocab_size, contexts_max_length)

    contexts_model.fit(X_train_contexts, y_train_contexts)
    predictions_train_contexts  = contexts_model.predict(X_train_contexts)
    predictions_test_contexts = contexts_model.predict(X_test_contexts)

    # Evaluate instance-level performance
    contexts_train_loss, contexts_train_acc, \
        contexts_train_auroc, contexts_train_aupr = contexts_model.evaluate_plus(X_train_contexts, \
                                                                                y_train_contexts, \
                                                                                predictions_train_contexts, \
                                                                                "contexts_predictions.instance.training")
    contexts_test_loss, contexts_test_acc, \
        contexts_test_auroc, contexts_test_aupr = contexts_model.evaluate_plus(X_test_contexts, \
                                                                                y_test_contexts, \
                                                                                predictions_test_contexts, \
                                                                                "contexts_predictions.instance.test")

    # Format output for display of results
    contexts_train_output_df = format_contexts_output(training_preprocessor.contexts_as_strings, \
                                                        training_preprocessor.context_seq_ids, \
                                                        training_preprocessor.context_ids, \
                                                        y_train_contexts, predictions_train_contexts, num_ctxt)
    contexts_test_output_df = format_contexts_output(test_preprocessor.contexts_as_strings, \
                                                        test_preprocessor.context_seq_ids, \
                                                        test_preprocessor.context_ids, \
                                                        y_test_contexts, predictions_test_contexts, num_ctxt)

    # Save results to file
    contexts_train_output_filename = "contexts_predictions.training.tsv"
    contexts_train_output_df.to_csv(contexts_train_output_filename, sep = "\t", header = False, index = False)
    contexts_test_output_filename = "contexts_predictions.test.tsv"
    contexts_test_output_df.to_csv(contexts_test_output_filename, sep = "\t", header = False, index = False)

    # Write contexts to a fasta file
    #test_preprocessor.write_ctxt_to_fasta(contexts_test_output_df, threshold)
    
    # Print summary of contexts results
    with open(summary_output_file, 'a') as f:
        print("----- Contexts model ----- ", file = f)
        print("\tTraining Accuracy:" + "\t" + str(contexts_train_acc), file = f)
        print("\tTraining AUROC:" + "\t" + str(contexts_train_auroc), file = f)
        print("\tTraining AUPR:" + "\t" + str(contexts_train_aupr), file = f)
        print("", file = f)
        print("\tTest Accuracy:" + "\t" + str(contexts_test_acc), file = f)
        print("\tTest AUROC:" + "\t" + str(contexts_test_auroc), file = f)
        print("\tTest AUPR:" + "\t" + str(contexts_test_aupr), file = f)
        print("", file = f)
        print("\tAvg CV Train AUROC:" + "\t" + str(contexts_CV_auroc_df.loc["avg"]["train"]) + " +/- " + \
                                                    str(contexts_CV_auroc_df.loc["stdev"]["train"]), file = f)
        print("\tAvg CV Test AUROC:" + "\t" + str(contexts_CV_auroc_df.loc["avg"]["test"]) + " +/- " + \
                                                    str(contexts_CV_auroc_df.loc["stdev"]["test"]), file = f)
        print("", file = f)
        print("----- Contexts model, bag performance ----- ", file = f)

    # Evaluate bag-level performance (methods can be max, avg, vote, and soft)
    # bag_performance_methods = ["max", "avg", "vote", "soft"]
    # bag_performance_methods = ["max", "avg", "vote"]
    bag_performance_methods = ["avg"]
    for method in bag_performance_methods:
        contexts_train_bag_acc, \
            contexts_train_bag_auroc, \
                contexts_train_bag_aupr = contexts_model.evaluate_bag(y_train_split_seq, contexts_train_output_df, threshold, \
                                                                        method, "contexts_predictions.bag." + method + ".training")
        contexts_test_bag_acc, \
            contexts_test_bag_auroc, \
                contexts_test_bag_aupr = contexts_model.evaluate_bag(y_test_split_seq, contexts_test_output_df, threshold, \
                                                                        method, "contexts_predictions.bag." + method + ".test")
        
        with open(summary_output_file, 'a') as f:
            print("\tTraining Bag Accuracy (" + method + "):" + "\t" + str(contexts_train_bag_acc), file = f)
            print("\tTraining Bag AUROC (" + method + "):" + "\t" + str(contexts_train_bag_auroc), file = f)
            print("\tTraining Bag AUPR (" + method + "):" + "\t" + str(contexts_train_bag_aupr), file = f)
            print("", file = f)
            print("\tTest Bag Accuracy (" + method + "):" + "\t" + str(contexts_test_bag_acc), file = f)
            print("\tTest Bag AUROC (" + method + "):" + "\t" + str(contexts_test_bag_auroc), file = f)
            print("\tTest Bag AUPR (" + method + "):" + "\t" + str(contexts_test_bag_aupr), file = f)
            print("", file = f)

        if (method == "max"):
            contexts_train_bag_acc_max = contexts_train_bag_acc
            contexts_train_bag_auroc_max = contexts_train_bag_auroc
            contexts_train_bag_aupr_max = contexts_train_bag_aupr
            contexts_test_bag_acc_max = contexts_test_bag_acc
            contexts_test_bag_auroc_max = contexts_test_bag_auroc
            contexts_test_bag_aupr_max = contexts_test_bag_aupr
        if (method == "avg"):
            contexts_train_bag_acc_avg = contexts_train_bag_acc
            contexts_train_bag_auroc_avg = contexts_train_bag_auroc
            contexts_train_bag_aupr_avg = contexts_train_bag_aupr
            contexts_test_bag_acc_avg = contexts_test_bag_acc
            contexts_test_bag_auroc_avg = contexts_test_bag_auroc
            contexts_test_bag_aupr_avg = contexts_test_bag_aupr
        if (method == "vote"):
            contexts_train_bag_acc_vote = contexts_train_bag_acc
            contexts_train_bag_auroc_vote = contexts_train_bag_auroc
            contexts_train_bag_aupr_vote = contexts_train_bag_aupr
            contexts_test_bag_acc_vote = contexts_test_bag_acc
            contexts_test_bag_auroc_vote = contexts_test_bag_auroc
            contexts_test_bag_aupr_vote = contexts_test_bag_aupr

    # # Save model
    # contexts_model_filename = "contexts_model.keras"
    # contexts_model.save_model(contexts_model_filename)

    # Save object
    contexts_model.save_object("contexts_model.obj")

    # # ------------------------------------------------------- SUMMARIZE ALL RESULTS -------------------------------------------------------

    # # Create a dataframe with all results
    # accuracy_df = pd.DataFrame({"baseline_train": [baseline_train_acc], \
    #                             "baseline_val": [baseline_CV_accuracy_df.loc["avg", "train"]], \
    #                             "baseline_test": [baseline_test_acc], \
    #                             "contexts_train": [contexts_train_acc], \
    #                             "contexts_val": [contexts_CV_accuracy_df.loc["avg", "train"]], 
    #                             "contexts_test": [contexts_test_acc], \
    #                             "contexts_bag_train_max": [contexts_train_bag_acc_max], \
    #                             "contexts_bag_test_max": [contexts_test_bag_acc_max], \
    #                             "contexts_bag_train_avg": [contexts_train_bag_acc_avg], \
    #                             "contexts_bag_test_avg": [contexts_test_bag_acc_avg], \
    #                             "contexts_bag_train_vote": [contexts_train_bag_acc_vote], \
    #                             "contexts_bag_test_vote": [contexts_test_bag_acc_vote]})

    # auroc_df = pd.DataFrame({"baseline_train": [baseline_train_auroc], \
    #                             "baseline_val": [baseline_CV_auroc_df.loc["avg", "train"]], \
    #                             "baseline_test": [baseline_test_auroc], \
    #                             "contexts_train": [contexts_train_auroc], \
    #                             "contexts_val": [contexts_CV_auroc_df.loc["avg", "train"]], 
    #                             "contexts_test": [contexts_test_auroc], \
    #                             "contexts_bag_train_max": [contexts_train_bag_auroc_max], \
    #                             "contexts_bag_test_max": [contexts_test_bag_auroc_max], \
    #                             "contexts_bag_train_avg": [contexts_train_bag_auroc_avg], \
    #                             "contexts_bag_test_avg": [contexts_test_bag_auroc_avg], \
    #                             "contexts_bag_train_vote": [contexts_train_bag_auroc_vote], \
    #                             "contexts_bag_test_vote": [contexts_test_bag_auroc_vote]})
    
    # aupr_df = pd.DataFrame({"baseline_train": [baseline_train_aupr], \
    #                             "baseline_val": [baseline_CV_aupr_df.loc["avg", "train"]], \
    #                             "baseline_test": [baseline_test_aupr], \
    #                             "contexts_train": [contexts_train_aupr], \
    #                             "contexts_val": [contexts_CV_aupr_df.loc["avg", "train"]], 
    #                             "contexts_test": [contexts_test_aupr], \
    #                             "contexts_bag_train_max": [contexts_train_bag_aupr_max], \
    #                             "contexts_bag_test_max": [contexts_test_bag_aupr_max], \
    #                             "contexts_bag_train_avg": [contexts_train_bag_aupr_avg], \
    #                             "contexts_bag_test_avg": [contexts_test_bag_aupr_avg], \
    #                             "contexts_bag_train_vote": [contexts_train_bag_aupr_vote], \
    #                             "contexts_bag_test_vote": [contexts_test_bag_aupr_vote]})

    # ------------------------------------------ RUN SPLIT SEQUENCE (BASELINE) MODEL ON CONTEXTS ------------------------------------------

    print("Running baseline model on contexts...")

    # Encode contexts using baseline model's tokenizer
    # Encode X as a count matrix (Bag-of-Words)
    X_train_baseline_on_contexts = baseline_model.encode_matrix(training_preprocessor.contexts_as_strings, baseline_mode)
    X_test_baseline_on_contexts = baseline_model.encode_matrix(test_preprocessor.contexts_as_strings, baseline_mode)

    # Format y
    y_train_baseline_on_contexts = np.asarray(training_preprocessor.context_labels) # length is total num sequences in training set
    y_test_baseline_on_contexts = np.asarray(test_preprocessor.context_labels) # length is total num sequences in test set

    # Save the baseline on contexts class distribution
    baseline_on_contexts_class_distribution = get_class_distribution_test(y_test_baseline_on_contexts)
    baseline_on_contexts_class_distribution.to_csv("baseline_on_contexts.class_distributions.tsv", sep = "\t", index = False)

    # Run baseline model on contexts
    predictions_train_baseline_on_contexts = baseline_model.predict(X_train_baseline_on_contexts)
    predictions_test_baseline_on_contexts = baseline_model.predict(X_test_baseline_on_contexts)

    # # Find how many predictions are positive and negative
    # num_predictions_test_baseline_on_contexts_positive = sum([x[0] > threshold for x in predictions_test_baseline_on_contexts])
    # num_predictions_test_baseline_on_contexts_negative = sum([x[0] < threshold for x in predictions_test_baseline_on_contexts])

    baseline_on_contexts_train_loss, \
        baseline_on_contexts_train_acc, \
            baseline_on_contexts_train_auroc, \
                baseline_on_contexts_train_aupr = baseline_model.evaluate_plus(X_train_baseline_on_contexts, \
                                                                                y_train_contexts, \
                                                                                predictions_train_baseline_on_contexts, \
                                                                                "baseline_on_contexts_predictions.instance.training")

    baseline_on_contexts_test_loss, \
        baseline_on_contexts_test_acc, \
            baseline_on_contexts_test_auroc, \
                baseline_on_contexts_test_aupr = baseline_model.evaluate_plus(X_test_baseline_on_contexts, \
                                                                                y_test_contexts, \
                                                                                predictions_test_baseline_on_contexts, \
                                                                                "baseline_on_contexts_predictions.instance.test")

    baseline_on_contexts_train_output_df = format_contexts_output(training_preprocessor.contexts_as_strings, \
                                                                    training_preprocessor.context_seq_ids, \
                                                                    training_preprocessor.context_ids, \
                                                                    y_train_baseline_on_contexts, \
                                                                    predictions_train_baseline_on_contexts, num_ctxt)
    baseline_on_contexts_test_output_df = format_contexts_output(test_preprocessor.contexts_as_strings, \
                                                                    test_preprocessor.context_seq_ids, \
                                                                    test_preprocessor.context_ids, \
                                                                    y_test_baseline_on_contexts, \
                                                                    predictions_test_baseline_on_contexts, num_ctxt)

    # Save results to file
    baseline_on_contexts_train_output_filename = "baseline_on_contexts_predictions.training.tsv"
    baseline_on_contexts_train_output_df.to_csv(baseline_on_contexts_train_output_filename, sep = "\t", header = False, index = False)
    baseline_on_contexts_test_output_file = "baseline_on_contexts_predictions.test.tsv"
    baseline_on_contexts_test_output_df.to_csv(baseline_on_contexts_test_output_file, sep = "\t", header = False, index = False)

    # Write contexts to a fasta file
    #test_preprocessor.write_baseline_on_ctxt_to_fasta(baseline_on_contexts_test_output_df, threshold)

    # Print summary of contexts results
    with open(summary_output_file, 'a') as f:
        print("----- Baseline on contexts ----- ", file = f)
        print("\tTraining Accuracy:" + "\t" + str(baseline_on_contexts_train_acc), file = f)
        print("\tTraining AUROC:" + "\t" + str(baseline_on_contexts_train_auroc), file = f)
        print("\tTraining AUPR:" + "\t" + str(baseline_on_contexts_train_aupr), file = f)
        print("", file = f)
        print("\tTest Accuracy:" + "\t" + str(baseline_on_contexts_test_acc), file = f)
        print("\tTest AUROC:" + "\t" + str(baseline_on_contexts_test_auroc), file = f)
        print("\tTest AUPR:" + "\t" + str(baseline_on_contexts_test_aupr), file = f)
        print("", file = f)
        print("----- Baseline on contexts, bag performance ----- ", file = f)

    # Evaluate bag-level performance (methods can be max, avg, vote, and soft)
    # bag_performance_methods = ["max", "avg", "vote"]
    bag_performance_methods = ["avg"]
    for method in bag_performance_methods:
        baseline_on_contexts_train_bag_acc, \
            baseline_on_contexts_train_bag_auroc, \
                baseline_on_contexts_train_bag_aupr = baseline_model.evaluate_bag(y_train_split_seq, \
                                                                                    baseline_on_contexts_train_output_df, \
                                                                                    threshold, method, \
                                                                                    "baseline_on_contexts_predictions.bag." + method + ".training")
        
        baseline_on_contexts_test_bag_acc, \
            baseline_on_contexts_test_bag_auroc, \
                baseline_on_contexts_test_bag_aupr = baseline_model.evaluate_bag(y_test_split_seq, \
                                                                                    baseline_on_contexts_test_output_df, \
                                                                                    threshold, method, \
                                                                                    "baseline_on_contexts_predictions.bag." + method + ".test")
        with open(summary_output_file, 'a') as f:
            print("\tTraining Bag Accuracy (" + method + "):" + "\t" + str(baseline_on_contexts_train_bag_acc), file = f)
            print("\tTraining Bag AUROC (" + method + "):" + "\t" + str(baseline_on_contexts_train_bag_auroc), file = f)
            print("\tTraining Bag AUPR (" + method + "):" + "\t" + str(baseline_on_contexts_train_bag_aupr), file = f)
            print("", file = f)
            print("\tTest Bag Accuracy (" + method + "):" + "\t" + str(baseline_on_contexts_test_bag_acc), file = f)
            print("\tTest Bag AUROC (" + method + "):" + "\t" + str(baseline_on_contexts_test_bag_auroc), file = f)
            print("\tTest Bag AUPR (" + method + "):" + "\t" + str(baseline_on_contexts_test_bag_aupr), file = f)
            print("", file = f)

        if (method == "max"):
            baseline_on_contexts_train_bag_acc_max = baseline_on_contexts_train_bag_acc
            baseline_on_contexts_train_bag_auroc_max = baseline_on_contexts_train_bag_auroc
            baseline_on_contexts_train_bag_aupr_max = baseline_on_contexts_train_bag_aupr
            baseline_on_contexts_test_bag_acc_max = baseline_on_contexts_test_bag_acc
            baseline_on_contexts_test_bag_auroc_max = baseline_on_contexts_test_bag_auroc
            baseline_on_contexts_test_bag_aupr_max = baseline_on_contexts_test_bag_aupr
        if (method == "avg"):
            baseline_on_contexts_train_bag_acc_avg = baseline_on_contexts_train_bag_acc
            baseline_on_contexts_train_bag_auroc_avg = baseline_on_contexts_train_bag_auroc
            baseline_on_contexts_train_bag_aupr_avg = baseline_on_contexts_train_bag_aupr
            baseline_on_contexts_test_bag_acc_avg = baseline_on_contexts_test_bag_acc
            baseline_on_contexts_test_bag_auroc_avg = baseline_on_contexts_test_bag_auroc
            baseline_on_contexts_test_bag_aupr_avg = baseline_on_contexts_test_bag_aupr
        if (method == "vote"):
            baseline_on_contexts_train_bag_acc_vote = baseline_on_contexts_train_bag_acc
            baseline_on_contexts_train_bag_auroc_vote = baseline_on_contexts_train_bag_auroc
            baseline_on_contexts_train_bag_aupr_vote = baseline_on_contexts_train_bag_aupr
            baseline_on_contexts_test_bag_acc_vote = baseline_on_contexts_test_bag_acc
            baseline_on_contexts_test_bag_auroc_vote = baseline_on_contexts_test_bag_auroc
            baseline_on_contexts_test_bag_aupr_vote = baseline_on_contexts_test_bag_aupr

     # ------------------------------------------------------- SUMMARIZE ALL RESULTS -------------------------------------------------------

    # Create a dataframe with all results
    accuracy_df = pd.DataFrame({"baseline_train": [baseline_train_acc], \
                                "baseline_val": [baseline_CV_accuracy_df.loc["avg", "train"]], \
                                "baseline_test": [baseline_test_acc], \
                                "contexts_train": [contexts_train_acc], \
                                "contexts_val": [contexts_CV_accuracy_df.loc["avg", "train"]], 
                                "contexts_test": [contexts_test_acc], \
                                # "contexts_bag_train_max": [contexts_train_bag_acc_max], \
                                # "contexts_bag_test_max": [contexts_test_bag_acc_max], \
                                "contexts_bag_train_avg": [contexts_train_bag_acc_avg], \
                                "contexts_bag_test_avg": [contexts_test_bag_acc_avg], \
                                # "contexts_bag_train_vote": [contexts_train_bag_acc_vote], \
                                # "contexts_bag_test_vote": [contexts_test_bag_acc_vote], \
                                "baseline_on_contexts_train": [baseline_on_contexts_train_acc], \
                                "baseline_on_contexts_test": [baseline_on_contexts_test_acc], \
                                # "baseline_on_contexts_bag_train_max": [baseline_on_contexts_train_bag_acc_max], \
                                # "baseline_on_contexts_bag_test_max": [baseline_on_contexts_test_bag_acc_max], \
                                "baseline_on_contexts_bag_train_avg": [baseline_on_contexts_train_bag_acc_avg], \
                                "baseline_on_contexts_bag_test_avg": [baseline_on_contexts_test_bag_acc_avg]}) # , \
                                # "baseline_on_contexts_bag_train_vote": [baseline_on_contexts_train_bag_acc_vote], \
                                # "baseline_on_contexts_bag_test_vote": [baseline_on_contexts_test_bag_acc_vote]})

    auroc_df = pd.DataFrame({"baseline_train": [baseline_train_auroc], \
                                "baseline_val": [baseline_CV_auroc_df.loc["avg", "train"]], \
                                "baseline_test": [baseline_test_auroc], \
                                "contexts_train": [contexts_train_auroc], \
                                "contexts_val": [contexts_CV_auroc_df.loc["avg", "train"]], 
                                "contexts_test": [contexts_test_auroc], \
                                # "contexts_bag_train_max": [contexts_train_bag_auroc_max], \
                                # "contexts_bag_test_max": [contexts_test_bag_auroc_max], \
                                "contexts_bag_train_avg": [contexts_train_bag_auroc_avg], \
                                "contexts_bag_test_avg": [contexts_test_bag_auroc_avg], \
                                # "contexts_bag_train_vote": [contexts_train_bag_auroc_vote], \
                                # "contexts_bag_test_vote": [contexts_test_bag_auroc_vote], \
                                "baseline_on_contexts_train": [baseline_on_contexts_train_auroc], \
                                "baseline_on_contexts_test": [baseline_on_contexts_test_auroc], \
                                # "baseline_on_contexts_bag_train_max": [baseline_on_contexts_train_bag_auroc_max], \
                                # "baseline_on_contexts_bag_test_max": [baseline_on_contexts_test_bag_auroc_max], \
                                "baseline_on_contexts_bag_train_avg": [baseline_on_contexts_train_bag_auroc_avg], \
                                "baseline_on_contexts_bag_test_avg": [baseline_on_contexts_test_bag_auroc_avg]}) # , \
                                # "baseline_on_contexts_bag_train_vote": [baseline_on_contexts_train_bag_auroc_vote], \
                                # "baseline_on_contexts_bag_test_vote": [baseline_on_contexts_test_bag_auroc_vote]})
    
    aupr_df = pd.DataFrame({"baseline_train": [baseline_train_aupr], \
                                "baseline_val": [baseline_CV_aupr_df.loc["avg", "train"]], \
                                "baseline_test": [baseline_test_aupr], \
                                "contexts_train": [contexts_train_aupr], \
                                "contexts_val": [contexts_CV_aupr_df.loc["avg", "train"]], 
                                "contexts_test": [contexts_test_aupr], \
                                # "contexts_bag_train_max": [contexts_train_bag_aupr_max], \
                                # "contexts_bag_test_max": [contexts_test_bag_aupr_max], \
                                "contexts_bag_train_avg": [contexts_train_bag_aupr_avg], \
                                "contexts_bag_test_avg": [contexts_test_bag_aupr_avg], \
                                # "contexts_bag_train_vote": [contexts_train_bag_aupr_vote], \
                                # "contexts_bag_test_vote": [contexts_test_bag_aupr_vote], \
                                "baseline_on_contexts_train": [baseline_on_contexts_train_aupr], \
                                "baseline_on_contexts_test": [baseline_on_contexts_test_aupr], \
                                # "baseline_on_contexts_bag_train_max": [baseline_on_contexts_train_bag_aupr_max], \
                                # "baseline_on_contexts_bag_test_max": [baseline_on_contexts_test_bag_aupr_max], \
                                "baseline_on_contexts_bag_train_avg": [baseline_on_contexts_train_bag_aupr_avg], \
                                "baseline_on_contexts_bag_test_avg": [baseline_on_contexts_test_bag_aupr_avg]}) # , \
                                # "baseline_on_contexts_bag_train_vote": [baseline_on_contexts_train_bag_aupr_vote], \
                                # "baseline_on_contexts_bag_test_vote": [baseline_on_contexts_test_bag_aupr_vote]})

    # # ---------------------------------------------- RELABEL CONTEXTS AND RETRAIN ----------------------------------------------

    # Relabel test contexts according to baseline predictions
    y_train_contexts_relabeled = prepUtils.relabel(threshold, predictions_train_baseline_on_contexts, y_train_contexts)
    y_test_contexts_relabeled = prepUtils.relabel(threshold, predictions_test_baseline_on_contexts, y_test_contexts)

    # Print summary of relabeling parameters
    with open(summary_output_file, 'a') as f:
        print("----- Relabeling parameters -----", file = f)
        print("\tThreshold:" + "\t" + str(threshold), file = f)
        print("\tNumber of iterations:" + "\t" + str(num_iterations), file = f)
        print("", file = f)

    # Relabel for specified number of iterations
    for i in range(0, num_iterations):

        print("Relabeling and retraining, iteration " + str(i+1) + "...")
        
        # Save the relabeled contexts class distribution
        contexts_relabeled_class_distribution = get_class_distribution_test(y_test_contexts_relabeled)
        contexts_relabeled_class_distribution.to_csv("contexts_relabeled.class_distributions.iteration" + str(i+1) + ".tsv", sep = "\t", index = False)

        # Initialize contexts model
        contexts_relabeled_model = classifier.Classifier()

        contexts_relabeled_model.in_shape = X_train_contexts.shape[1]
        contexts_relabeled_model.hidden_shape = contexts_hidden_shape
        contexts_relabeled_model.out_shape = 1
        contexts_relabeled_model.embedding_dim = contexts_embedding_dim
        contexts_relabeled_model.num_filters = contexts_num_filters
        contexts_relabeled_model.kernel_length = contexts_kernel_length
        contexts_relabeled_model.kernel_stride = 1
        contexts_relabeled_model.pool_size = 2
        contexts_relabeled_model.epochs = contexts_epochs
        contexts_relabeled_model.batch_size = contexts_batch_size

        # dense model with BoW encoding
        if (contexts_model_type == "dense"):
            contexts_relabeled_model.build()
        # dense model with embedding layer
        elif (contexts_model_type == "dense_embed"):
            contexts_relabeled_model.build_w_Embedding(contexts_vocab_size, contexts_max_length)
        # cnn with embedding layer
        elif (contexts_model_type == "cnn"):
            contexts_relabeled_model.build_w_CNN(contexts_vocab_size, contexts_max_length)
        # cnn + lstm with embedding layer
        elif (contexts_model_type == "cnn_lstm"):
            contexts_relabeled_model.build_w_CNN_LSTM(contexts_vocab_size, contexts_max_length)
        # cnn + bilstm with embedding layer
        elif (contexts_model_type == "cnn_bilstm"):
            contexts_relabeled_model.build_w_CNN_biLSTM(contexts_vocab_size, contexts_max_length)
        # cnn + gru with embedding layer
        elif (contexts_model_type == "cnn_gru"):
            contexts_relabeled_model.build_w_CNN_GRU(contexts_vocab_size, contexts_max_length)

        # Refit contexts model with new labels
        contexts_relabeled_model.fit(X_train_contexts, y_train_contexts_relabeled)

        # Make predictions with re-trained model
        predictions_train_contexts_relabeled = contexts_relabeled_model.predict(X_train_contexts)
        predictions_test_contexts_relabeled = contexts_relabeled_model.predict(X_test_contexts)

        # # Find how many predictions are positive and negative
        # num_predictions_test_contexts_relabeled_positive = sum([x[0] > threshold for x in predictions_test_contexts_relabeled])
        # num_predictions_test_contexts_relabeled_negative = sum([x[0] < threshold for x in predictions_test_contexts_relabeled])

        # Evaluate instance-level performance
        contexts_relabeled_train_loss, \
            contexts_relabeled_train_acc, \
                contexts_relabeled_train_auroc, \
                    contexts_relabeled_train_aupr = contexts_relabeled_model.evaluate_plus(X_train_contexts, \
                                                                                    y_train_contexts_relabeled, \
                                                                                        predictions_train_contexts_relabeled, \
                                                                                            "contexts_relabeled_predictions.iteration" + str(i + 1) + ".instance.training")
        contexts_relabeled_test_loss, \
            contexts_relabeled_test_acc, \
                contexts_relabeled_test_auroc, \
                    contexts_relabeled_test_aupr = contexts_relabeled_model.evaluate_plus(X_test_contexts, \
                                                                                    y_test_contexts_relabeled, 
                                                                                        predictions_test_contexts_relabeled, \
                                                                                            "contexts_relabeled_predictions.iteration" + str(i + 1) + ".instance.test")

        # Format output for display of results
        contexts_relabeled_train_output_df = format_contexts_output(training_preprocessor.contexts_as_strings, \
                                                                    training_preprocessor.context_seq_ids, \
                                                                    training_preprocessor.context_ids, \
                                                                    y_train_contexts_relabeled, \
                                                                    predictions_train_contexts_relabeled, num_ctxt)
        contexts_relabeled_test_output_df = format_contexts_output(test_preprocessor.contexts_as_strings, \
                                                                    test_preprocessor.context_seq_ids, \
                                                                    test_preprocessor.context_ids, \
                                                                    y_test_contexts_relabeled, \
                                                                    predictions_test_contexts_relabeled, num_ctxt)
        # Save results to a file
        contexts_relabeled_train_output_filename = "contexts_relabeled_predictions.iteration" + str(i+1) + ".training.tsv"
        contexts_relabeled_train_output_df.to_csv(contexts_relabeled_train_output_filename, sep = "\t", header = False, index = False)
        contexts_relabeled_test_output_filename = "contexts_relabeled_predictions.iteration" + str(i+1) + ".test.tsv"
        contexts_relabeled_test_output_df.to_csv(contexts_relabeled_test_output_filename, sep = "\t", header = False, index = False)

        # Write contexts to a fasta file
        #test_preprocessor.write_relabeled_ctxt_to_fasta(contexts_relabeled_test_output_df, threshold, (i+1))

        # Print summary of baseline on contexts results
        with open(summary_output_file, 'a') as f:
            print("----- Contexts model (relabeled + retrained, iteration " + str(i+1) + ") -----", file = f)
            print("\tTraining Accuracy:" + "\t" + str(contexts_relabeled_train_acc), file = f)
            print("\tTraining AUROC:" + "\t" + str(contexts_relabeled_train_auroc), file = f)
            print("\tTraining AUPR:" + "\t" + str(contexts_relabeled_train_aupr), file = f)
            print("", file = f)
            print("\tTest Accuracy:" + "\t" + str(contexts_relabeled_test_acc), file = f)
            print("\tTest AUROC:" + "\t" + str(contexts_relabeled_test_auroc), file = f)
            print("\tTest AUPR:" + "\t" + str(contexts_relabeled_test_aupr), file = f)
            print("", file = f)
            print("----- Contexts model (relabeled + retrained, iteration " + str(i+1) + ", bag performance) ----- ", file = f)

        # Evaluate bag-level performance (methods can be max, avg, vote, and soft)
        # bag_performance_methods = ["max", "avg", "vote", "soft"]
        # bag_performance_methods = ["max", "avg", "vote"]
        bag_performance_methods = ["avg"]
        for method in bag_performance_methods:
            contexts_relabeled_train_bag_acc, \
                contexts_relabeled_train_bag_auroc, \
                    contexts_relabeled_train_bag_aupr = contexts_relabeled_model.evaluate_bag(y_train_split_seq, contexts_relabeled_train_output_df, \
                                                                                    threshold, method, \
                                                                                    "contexts_predictions.bag_performance.iteration" + str(i+1) + ".training." + method)
            contexts_relabeled_test_bag_acc, \
                contexts_relabeled_test_bag_auroc, \
                    contexts_relabeled_test_bag_aupr = contexts_relabeled_model.evaluate_bag(y_test_split_seq, contexts_relabeled_test_output_df, \
                                                                                    threshold, method, \
                                                                                    "contexts_predictions.bag_performance.iteration" + str(i+1) + ".test." + method)
            
            with open(summary_output_file, 'a') as f:
                print("\tTraining Bag Accuracy (" + method + "):" + "\t" + str(contexts_relabeled_train_bag_acc), file = f)
                print("\tTraining Bag AUROC (" + method + "):" + "\t" + str(contexts_relabeled_train_bag_auroc), file = f)
                print("\tTraining Bag AUPR (" + method + "):" + "\t" + str(contexts_relabeled_train_bag_aupr), file = f)
                print("", file = f)
                print("\tTest Bag Accuracy (" + method + "):" + "\t" + str(contexts_relabeled_test_bag_acc), file = f)
                print("\tTest Bag AUROC (" + method + "):" + "\t" + str(contexts_relabeled_test_bag_auroc), file = f)
                print("\tTest Bag AUPR (" + method + "):" + "\t" + str(contexts_relabeled_test_bag_aupr), file = f)
                print("", file = f)

            if (method == "max"):
                contexts_relabeled_train_bag_acc_max = contexts_relabeled_train_bag_acc
                contexts_relabeled_train_bag_auroc_max = contexts_relabeled_train_bag_auroc
                contexts_relabeled_train_bag_aupr_max = contexts_relabeled_train_bag_aupr
                contexts_relabeled_test_bag_acc_max = contexts_relabeled_test_bag_acc
                contexts_relabeled_test_bag_auroc_max = contexts_relabeled_test_bag_auroc
                contexts_relabeled_test_bag_aupr_max = contexts_relabeled_test_bag_aupr
            if (method == "avg"):
                contexts_relabeled_train_bag_acc_avg = contexts_relabeled_train_bag_acc
                contexts_relabeled_train_bag_auroc_avg = contexts_relabeled_train_bag_auroc
                contexts_relabeled_train_bag_aupr_avg = contexts_relabeled_train_bag_aupr
                contexts_relabeled_test_bag_acc_avg = contexts_relabeled_test_bag_acc
                contexts_relabeled_test_bag_auroc_avg = contexts_relabeled_test_bag_auroc
                contexts_relabeled_test_bag_aupr_avg = contexts_relabeled_test_bag_aupr
            if (method == "vote"):
                contexts_relabeled_train_bag_acc_vote = contexts_relabeled_train_bag_acc
                contexts_relabeled_train_bag_auroc_vote = contexts_relabeled_train_bag_auroc
                contexts_relabeled_train_bag_aupr_vote = contexts_relabeled_train_bag_aupr
                contexts_relabeled_test_bag_acc_vote = contexts_relabeled_test_bag_acc
                contexts_relabeled_test_bag_auroc_vote = contexts_relabeled_test_bag_auroc
                contexts_relabeled_test_bag_aupr_vote = contexts_relabeled_test_bag_aupr

        # # Save the current iteration of the model to a file
        # contexts_relabeled_model_filename = "contexts_model.iteration" + str(i + 1) + ".keras"
        # contexts_relabeled_model.save_model(contexts_relabeled_model_filename)

        # Save object
        contexts_relabeled_model.save_object("contexts_model.iteration" + str(i + 1) + ".obj")

        # Add performance for this iteration to the data frame
        accuracy_df = pd.concat([accuracy_df, \
                                pd.DataFrame({"contexts_relabeled_iteration" + str(i + 1) + "_train": [contexts_relabeled_train_acc], \
                                                "contexts_relabeled_iteration" + str(i + 1) + "_test": [contexts_relabeled_test_acc], \
                                                # "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_max": [contexts_relabeled_train_bag_acc_max], \
                                                # "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_max": [contexts_relabeled_test_bag_acc_max], \
                                                "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_avg": [contexts_relabeled_train_bag_acc_avg], \
                                                "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_avg": [contexts_relabeled_test_bag_acc_avg]})], \
                                                # "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_vote": [contexts_relabeled_train_bag_acc_vote], \
                                                # "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_vote": [contexts_relabeled_test_bag_acc_vote]})], \
                                axis = 1)
        
        auroc_df = pd.concat([auroc_df, \
                                pd.DataFrame({"contexts_relabeled_iteration" + str(i + 1) + "_train": [contexts_relabeled_train_auroc], \
                                             "contexts_relabeled_iteration" + str(i + 1) + "_test": [contexts_relabeled_test_auroc], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_max": [contexts_relabeled_train_bag_auroc_max], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_max": [contexts_relabeled_test_bag_auroc_max], \
                                             "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_avg": [contexts_relabeled_train_bag_auroc_avg], \
                                             "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_avg": [contexts_relabeled_test_bag_auroc_avg]})], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_vote": [contexts_relabeled_train_bag_auroc_vote], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_vote": [contexts_relabeled_test_bag_auroc_vote]})], \
                                axis = 1)
        
        aupr_df = pd.concat([aupr_df, \
                                pd.DataFrame({"contexts_relabeled_iteration" + str(i + 1) + "_train": [contexts_relabeled_train_aupr], \
                                             "contexts_relabeled_iteration" + str(i + 1) + "_test": [contexts_relabeled_test_aupr], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_max": [contexts_relabeled_train_bag_aupr_max], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_max": [contexts_relabeled_test_bag_aupr_max], \
                                             "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_avg": [contexts_relabeled_train_bag_aupr_avg], \
                                             "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_avg": [contexts_relabeled_test_bag_aupr_avg]})], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_train_vote": [contexts_relabeled_train_bag_aupr_vote], \
                                             # "contexts_relabeled_iteration" + str(i + 1) + "_bag_test_vote": [contexts_relabeled_test_bag_aupr_vote]})], \
                                axis = 1)
        

        # Relabel again
        y_train_contexts_relabeled = prepUtils.relabel(threshold, predictions_train_contexts_relabeled, y_train_contexts_relabeled)
        y_test_contexts_relabeled = prepUtils.relabel(threshold, predictions_test_contexts_relabeled, y_test_contexts_relabeled)

    print("Saving results...")

    # Save all objects
    training_preprocessor.save_object("training_preprocessor.obj")
    test_preprocessor.save_object("test_preprocessor.obj")
    chromCV.save_object("chromCV.obj")
    
    # Save performance summary
    accuracy_df.transpose().to_csv("performance_summary.accuracy.tsv", sep = "\t", header = False)
    auroc_df.transpose().to_csv("performance_summary.auroc.tsv", sep = "\t", header = False)
    aupr_df.transpose().to_csv("performance_summary.aupr.tsv", sep = "\t", header = False)




