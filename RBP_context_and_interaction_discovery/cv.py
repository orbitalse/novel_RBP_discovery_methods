# Shaimae Elhajjajy
# July 13, 2022
# Define ChromCV class for import to other scripts

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

import numpy as np
import pandas as pd
import pickle
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
import subprocess
from tensorflow import keras

import readData

# Class : ChromCV
# Description : class to split data into training and test
class ChromCV:

    def __init__(self, all_cv_groups, training_cv_groups = None, test_cv_group = None):
        self.all_cv_groups = all_cv_groups # List of all CV groups from chromCV
        self.training_seq_df = None # Sequences from training CV groups
        self.test_seq_df = None # Sequences from test CV groups
        self.chromCV_iterable_baseline = [] # List of tuples containing indices for training and validation sequences for each iteration of chromCV
        self.chromCV_iterable_contexts = [] # List of tuples containing indices for training and validation contexts for each iteration of chromCV

        if (training_cv_groups and test_cv_group):
            self.training_cv_groups = training_cv_groups
            self.test_cv_group = test_cv_group
        else:
            self.training_cv_groups = [] # List of CV groups for training
            self.test_cv_group = [] # List of CV group for testing (the largest CV group)
        
    
    # Function : get_largest_cv_group
    # Description : Find the test CV group, which will be the largest CV group with the most sequences
    # Parameters : 
    # - all_seq_df : dataframe containing all sequences and their properties
    # Returns : 
    # - largest_cv_group : the CV group to use for testing
    def get_largest_cv_group(self, all_seq_df):
        max_size = 0
        largest_cv_group = "0"
        for k in self.all_cv_groups:
            cv_group_size = len(all_seq_df.index[all_seq_df["CV_group"] == k].tolist())
            if (cv_group_size > max_size):
                max_size = cv_group_size
                largest_cv_group = k

        return(largest_cv_group)
    
    # Function : get_test_cv_group
    # Description : Assign largest CV group as test CV group
    # Parameters : 
    # - all_seq_df : dataframe containing all sequences and their properties 
    # Returns : 
    # - test_cv_group : the CV group to use for testing
    def get_test_cv_group(self, all_seq_df):
        test_cv_group = self.get_largest_cv_group(all_seq_df)
        
        return(test_cv_group)

    # Function : get_training_cv_groups
    # Description : Find the training CV groups, which will be everything except the test (largest) CV group
    # Parameters : 
    # - NA
    # Returns : 
    # training_cv_groups : list of CV groups to use for training
    def get_training_cv_groups(self):
        training_cv_groups = self.all_cv_groups.copy()
        training_cv_groups.remove(self.test_cv_group)

        return(training_cv_groups)
    
    # Function : get_test_cv_seq
    # Description : Get sequences belonging to the test CV group
    # Parameters : 
    # - all_seq_df : dataframe containing all sequences and their properties
    # Returns : 
    # - test_seq_df : dataframe containing all test sequences and their properties
    def get_test_cv_seq(self, all_seq_df):
        if (len(self.test_cv_group) == 0):
            self.get_test_cv_group(all_seq_df)
        test_seq_df = all_seq_df[all_seq_df["CV_group"] == self.test_cv_group]
        test_seq_df.reset_index(inplace = True, drop = True)
        
        return(test_seq_df)

    # Function : get_training_cv_seq
    # Description : Get sequences belonging to the training CV groups
    # Parameters : 
    # - all_seq_df : dataframe containing all sequences and their properties
    # Returns :
    # - training_seq_df : dataframe containing all training sequences and their properties
    def get_training_cv_seq(self, all_seq_df):
        if (len(self.test_cv_group) == 0):
            self.get_test_cv_group(all_seq_df)
        training_seq_df = all_seq_df[all_seq_df["CV_group"] != self.test_cv_group]
        training_seq_df.reset_index(inplace = True, drop = True)
        
        return(training_seq_df)

    # Function : split_train_test_CV
    # Description : Split all sequences into training and test datasets based on their CV group
    # Parameters : 
    # - all_seq_df : dataframe containing all sequences and their properties
    # Returns : 
    # - self.training_seq_df : dataframe containing all training sequences and their properties
    # - self.test_seq_df : dataframe containing all test sequences and their properties
    def split_train_test_CV(self, all_seq_df):
        self.test_cv_group = self.get_test_cv_group(all_seq_df)
        self.training_cv_groups = self.get_training_cv_groups() # Not used elsewhere, but want to fill attribute

        self.test_seq_df = self.get_test_cv_seq(all_seq_df)
        self.training_seq_df = self.get_training_cv_seq(all_seq_df)
        
        return(self.training_seq_df, self.test_seq_df)

    # Function : get_train_indices_baseline
    # Description : Given a CV group, get indices for training sequences for each CV fold
    # Parameters :
    # - CV_group : a CV group
    # - training_preprocessor : a prepUtils object that contains a list mapping each baseline sequence to its CV group
    # Returns : 
    # - train_indices : a list of indices for training sequences
    def get_train_indices_baseline(self, CV_group, training_preprocessor):
        train_indices = list(np.where(np.array(training_preprocessor.split_seq_cv_groups) != CV_group)[0])
        return(train_indices)
    
    # Function : get_validation_indices_baseline
    # Description : Given a CV group, get indices for validation sequences for each CV fold
    # Parameters :
    # - CV_group : a CV group
    # - training_preprocessor : a prepUtils object that contains a list mapping each baseline sequence to its CV group
    # Returns : 
    # - val_indices : a list of indices for validation sequences
    def get_validation_indices_baseline(self, CV_group, training_preprocessor):
        val_indices = list(np.where(np.array(training_preprocessor.split_seq_cv_groups) == CV_group)[0])
        return(val_indices)

    # Function : format_CV_iterable_baseline
    # Description : Format chromCV groups using indices of training and validation samples for each iteration
    # Parameters :
    # - NA
    # - training_preprocessor : a prepUtils object that contains a list mapping each baseline sequence to its CV group
    # Returns : 
    # - chromCV_iterable_baseline : tuple containing indices of training and validation observations for each iteration
    def format_CV_iterable_baseline(self, training_preprocessor):
        # Reinitialize the iterable if it has already been filled
        if len(self.chromCV_iterable_baseline) != 0:
            self.chromCV_iterable_baseline = []
        for CV_group in self.training_cv_groups:
            # Get indices for observations belonging to validation group
            val_indices = self.get_validation_indices_baseline(CV_group, training_preprocessor)
            # Get all other indices for observations belonging to training group
            train_indices = self.get_train_indices_baseline(CV_group, training_preprocessor)
            # Format indices for train and validation sets from this chromCV fold into a tuple
            self.chromCV_iterable_baseline.append((train_indices, val_indices))
        return(self.chromCV_iterable_baseline)
    
    # Function : get_train_indices_contexts
    # Description : Given a CV group, get indices for training sequences for each CV fold
    # Parameters :
    # - CV_group : a CV group
    # - training_preprocessor : a prepUtils object that contains a list mapping each context to its CV group
    # Returns : 
    # - train_indices : a list of indices for training sequences
    def get_train_indices_contexts(self, CV_group, training_preprocessor):
        train_indices = list(np.where(np.array(training_preprocessor.context_cv_groups) != CV_group)[0])
        return(train_indices)
    
    # Function : get_validation_indices_contexts
    # Description : Given a CV group, get indices for validation sequences for each CV fold
    # Parameters :
    # - CV_group : a CV group
    # - training_preprocessor : a prepUtils object that contains a list mapping each context to its CV group
    # Returns : 
    # - val_indices : a list of indices for validation sequences
    def get_validation_indices_contexts(self, CV_group, training_preprocessor):
        val_indices = list(np.where(np.array(training_preprocessor.context_cv_groups) == CV_group)[0])
        return(val_indices)

    # Function : format_CV_iterable_contexts
    # Description : Format chromCV groups using indices of training and validation samples for each iteration
    # Parameters :
    # - NA
    # - training_preprocessor : a prepUtils object that contains a list mapping each context to its CV group
    # Returns : 
    # - chromCV_iterable_contexts : tuple containing indices of training and validation observations for each iteration
    def format_CV_iterable_contexts(self, training_preprocessor):
        # Reinitialize the iterable if it has already been filled
        if len(self.chromCV_iterable_contexts) != 0:
            self.chromCV_iterable_contexts = []
        for CV_group in self.training_cv_groups:
            # Get indices for observations belonging to validation group
            val_indices = self.get_validation_indices_contexts(CV_group, training_preprocessor)
            # Get all other indices for observations belonging to training group
            train_indices = self.get_train_indices_contexts(CV_group, training_preprocessor)
            # Format indices for train and validation sets from this chromCV fold into a tuple
            self.chromCV_iterable_contexts.append((train_indices, val_indices))
        return(self.chromCV_iterable_contexts)

    # Save the cv object
    def save_object(self, filename):
        filehandler = open(filename, "wb")
        pickle.dump(self, filehandler)
        filehandler.close()
        return()

# Versions of model builds from classifier.py, re-written to work with Keras wrapper
def build_BoW(hidden_shape, in_shape, out_shape, dropout, dropout_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units = hidden_shape, \
                                    input_shape = (in_shape,), \
                                    activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
    if (dropout == True):
        model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(units = out_shape, \
                                    activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
    model.compile(loss = "binary_crossentropy", \
                    optimizer = "adam", \
                    metrics = ["accuracy", "AUC"]) # loss function for binary classification
    model.summary()
    return(model)

def build_w_Embedding(hidden_shape, in_shape, out_shape, vocab_size, embedding_dim, max_length, dropout, dropout_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = hidden_shape, \
                                    activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
    if (dropout == True):
        model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(units = out_shape, \
                                    activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", \
                    optimizer = "adam", \
                    metrics = ["accuracy", "AUC"])
    model.summary()
    return(model)

def build_wo_Embedding(hidden_shape, in_shape, out_shape, vocab_size, embedding_dim, max_length, dropout, dropout_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = (max_length, embedding_dim,)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = hidden_shape, \
                                    activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
    if (dropout == True):
        model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(units = out_shape, \
                                    activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
    model.compile(loss = "binary_crossentropy", \
                        optimizer = "adam", \
                        metrics = ["accuracy", "AUC"]) # loss function for binary classification
    model.summary()
    return(model)

def build_w_CNN(hidden_shape, in_shape, out_shape, vocab_size, embedding_dim, max_length, \
                num_filters, kernel_length, kernel_stride, pool_size, dropout, dropout_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length))
    model.add(keras.layers.Conv1D(filters = num_filters, \
                                            kernel_size = kernel_length, \
                                            strides = kernel_stride, \
                                            activation = "relu"))
    model.add(keras.layers.MaxPooling1D(pool_size = pool_size))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = hidden_shape, \
                                    activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
    if (dropout == True):
        model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(units = out_shape, \
                                    activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", \
                    optimizer = "adam", \
                    metrics = ["accuracy", "AUC"])
    model.summary()
    return(model)

def build_w_CNN_LSTM(hidden_shape, in_shape, out_shape, vocab_size, embedding_dim, max_length, \
                        num_filters, kernel_length, kernel_stride, pool_size, dropout, dropout_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length))
    model.add(keras.layers.Conv1D(filters = num_filters, \
                                            kernel_size = kernel_length, \
                                            strides = kernel_stride, \
                                            activation = "relu"))
    model.add(keras.layers.MaxPooling1D(pool_size = pool_size))
    model.add(keras.layers.LSTM(units = 64, dropout = 0.1, recurrent_dropout = 0.5))
    model.add(keras.layers.Dense(units = hidden_shape, \
                                    activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
    if (dropout == True):
        model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(units = out_shape, \
                                    activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", \
                    optimizer = "adam", \
                    metrics = ["accuracy", "AUC"])
    model.summary()
    return(model)

def build_w_CNN_biLSTM(hidden_shape, in_shape, out_shape, vocab_size, embedding_dim, max_length, \
                        num_filters, kernel_length, kernel_stride, pool_size, dropout, dropout_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length))
    model.add(keras.layers.Conv1D(filters = num_filters, \
                                            kernel_size = kernel_length, \
                                            strides = kernel_stride, \
                                            activation = "relu"))
    model.add(keras.layers.MaxPooling1D(pool_size = pool_size))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = 64, dropout = 0.1, recurrent_dropout = 0.5)))
    model.add(keras.layers.Dense(units = hidden_shape, \
                                    activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
    if (dropout == True):
        model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(units = out_shape, \
                                    activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", \
                    optimizer = "adam", \
                    metrics = ["accuracy", "AUC"])
    model.summary()
    return(model)

def build_w_CNN_GRU(hidden_shape, in_shape, out_shape, vocab_size, embedding_dim, max_length, \
                        num_filters, kernel_length, kernel_stride, pool_size, dropout, dropout_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length))
    model.add(keras.layers.Conv1D(filters = num_filters, \
                                            kernel_size = kernel_length, \
                                            strides = kernel_stride, \
                                            activation = "relu"))
    model.add(keras.layers.MaxPooling1D(pool_size = pool_size))
    model.add(keras.layers.GRU(units = 64, dropout = 0.1, recurrent_dropout = 0.5))
    model.add(keras.layers.Dense(units = hidden_shape, \
                                    activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
    if (dropout == True):
        model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(units = out_shape, \
                                    activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", \
                    optimizer = "adam", \
                    metrics = ["accuracy", "AUC"])
    model.summary()
    return(model)

# Function : build_model_wrapper
def build_model_wrapper(model_type, model_obj, vocab_size, max_length):
    # dense model with BoW encoding
    if (model_type == "dense"):
        model_wrapper = KerasClassifier(
            build_fn = build_BoW, # function to initialize and build keras model
            in_shape = model_obj.in_shape, # size of input (= size of vocab)
            hidden_shape = model_obj.hidden_shape, # default number of neurons        
            out_shape = model_obj.out_shape, # size of output (1 b/c binary classification)
            random_state = 42, # random seed for reproducibility
            dropout_rate = model_obj.dropout_rate, # percentage of neurons to drop
            dropout = model_obj.dropout # whether or not to add the dropout layer
        )
    # dense model with embedding layer
    elif (model_type == "dense_embed"):
        model_wrapper = KerasClassifier(
            build_fn = build_w_Embedding, # function to initialize and build keras model
            in_shape = model_obj.in_shape, # size of input (= size of vocab)
            hidden_shape = model_obj.hidden_shape, # default number of neurons        
            out_shape = model_obj.out_shape, # size of output (1 b/c binary classification)
            random_state = 42, # random seed for reproducibility
            vocab_size = vocab_size,
            embedding_dim = model_obj.embedding_dim,
            max_length = max_length,
            dropout_rate = model_obj.dropout_rate, # percentage of neurons to drop
            dropout = model_obj.dropout # whether or not to add the dropout layer
        )
    # dense model with separate embedding
    elif (model_type == "dense_embed_separate"):
        model_wrapper = KerasClassifier(
            build_fn = build_wo_Embedding, # function to initialize and build keras model
            in_shape = model_obj.in_shape, # size of input (= size of vocab)
            hidden_shape = model_obj.hidden_shape, # default number of neurons        
            out_shape = model_obj.out_shape, # size of output (1 b/c binary classification)
            random_state = 42, # random seed for reproducibility
            vocab_size = vocab_size,
            embedding_dim = model_obj.embedding_dim,
            max_length = max_length,
            dropout_rate = model_obj.dropout_rate, # percentage of neurons to drop
            dropout = model_obj.dropout # whether or not to add the dropout layer
        )
    # cnn with embedding layer
    elif (model_type == "cnn"):
        model_wrapper = KerasClassifier(
            build_fn = build_w_CNN, # function to initialize and build keras model
            in_shape = model_obj.in_shape, # size of input (= size of vocab)
            hidden_shape = model_obj.hidden_shape, # default number of neurons        
            out_shape = model_obj.out_shape, # size of output (1 b/c binary classification)
            random_state = 42, # random seed for reproducibility
            vocab_size = vocab_size,
            embedding_dim = model_obj.embedding_dim,
            max_length = max_length,
            num_filters = model_obj.num_filters,
            kernel_length = model_obj.kernel_length,
            kernel_stride = model_obj.kernel_stride,
            pool_size = model_obj.pool_size,
            dropout_rate = model_obj.dropout_rate, # percentage of neurons to drop
            dropout = model_obj.dropout # whether or not to add the dropout layer
        )
    # cnn + lstm with embedding layer
    elif (model_type == "cnn_lstm"):
        model_wrapper = KerasClassifier(
            build_fn = build_w_CNN_LSTM, # function to initialize and build keras model
            in_shape = model_obj.in_shape, # size of input (= size of vocab)
            hidden_shape = model_obj.hidden_shape, # default number of neurons        
            out_shape = model_obj.out_shape, # size of output (1 b/c binary classification)
            random_state = 42, # random seed for reproducibility
            vocab_size = vocab_size,
            embedding_dim = model_obj.embedding_dim,
            max_length = max_length,
            num_filters = model_obj.num_filters,
            kernel_length = model_obj.kernel_length,
            kernel_stride = model_obj.kernel_stride,
            pool_size = model_obj.pool_size,
            dropout_rate = model_obj.dropout_rate, # percentage of neurons to drop
            dropout = model_obj.dropout # whether or not to add the dropout layer
        )
    # cnn + bilstm with embedding layer
    elif (model_type == "cnn_bilstm"):
        model_wrapper = KerasClassifier(
            build_fn = build_w_CNN_biLSTM, # function to initialize and build keras model
            in_shape = model_obj.in_shape, # size of input (= size of vocab)
            hidden_shape = model_obj.hidden_shape, # default number of neurons        
            out_shape = model_obj.out_shape, # size of output (1 b/c binary classification)
            random_state = 42, # random seed for reproducibility
            vocab_size = vocab_size,
            embedding_dim = model_obj.embedding_dim,
            max_length = max_length,
            num_filters = model_obj.num_filters,
            kernel_length = model_obj.kernel_length,
            kernel_stride = model_obj.kernel_stride,
            pool_size = model_obj.pool_size,
            dropout_rate = model_obj.dropout_rate, # percentage of neurons to drop
            dropout = model_obj.dropout # whether or not to add the dropout layer
        )
    # cnn + gru with embedding layer
    elif (model_type == "cnn_gru"):
        model_wrapper = KerasClassifier(
            build_fn = build_w_CNN_GRU, # function to initialize and build keras model
            in_shape = model_obj.in_shape, # size of input (= size of vocab)
            hidden_shape = model_obj.hidden_shape, # default number of neurons        
            out_shape = model_obj.out_shape, # size of output (1 b/c binary classification)
            random_state = 42, # random seed for reproducibility
            vocab_size = vocab_size,
            embedding_dim = model_obj.embedding_dim,
            max_length = max_length,
            num_filters = model_obj.num_filters,
            kernel_length = model_obj.kernel_length,
            kernel_stride = model_obj.kernel_stride,
            pool_size = model_obj.pool_size,
            dropout_rate = model_obj.dropout_rate, # percentage of neurons to drop
            dropout = model_obj.dropout # whether or not to add the dropout layer
        )
    return(model_wrapper)
        
# Function : run_CV
# Description : Given a range of values and number of iterations, perform randomized hyperparameter tuning 
# Note : This also includes cross validation, using custom user-defined folds
# Parameters : 
# - model_wrapper : 
# - model_hyperparams : a dictionary specifying the range of hyperparameter values to sample from
# - chromCV_iterable : 
# - scoring_function : the custom function to score models trained with each combination of hyperparameters
# - X_train : the training data
# - X_test : the test data
# Results :
# - search : the results of the randomized hyperparameter tuning
def run_CV(model_type, model_obj, vocab_size, max_length, \
            model_hyperparams, chromCV_iterable, custom_scoring_function, \
                X_train, y_train):
    model_wrapper = build_model_wrapper(model_type, model_obj, vocab_size, max_length)
    search = RandomizedSearchCV(model_wrapper, \
                                param_distributions = model_hyperparams, \
                                n_iter = 25, \
                                random_state = 42, \
                                return_train_score = True, \
                                scoring = custom_scoring_function, \
                                n_jobs = 1, \
                                cv = chromCV_iterable, \
                                verbose = 3, \
                                refit = "auroc")
    search.fit(X_train, y_train)
    return(search)

def compute_acc(y_true, y_pred):
    acc_obj = keras.metrics.BinaryAccuracy()
    acc_obj.update_state(y_true, y_pred)
    acc = acc_obj.result().numpy()
    return(acc)

def compute_auroc(y_true, y_pred):
    roc_curve = keras.metrics.AUC(curve = "ROC")
    roc_curve.update_state(y_true, y_pred)
    auc = roc_curve.result().numpy()
    return(auc)

def compute_aupr(y_true, y_pred):
    pr_curve = keras.metrics.AUC(curve = "PR")
    pr_curve.update_state(y_true, y_pred)
    aupr = pr_curve.result().numpy()
    return(aupr)

def evaluate(y_true, y_pred):
    acc = compute_acc(y_true, y_pred)
    auroc = compute_auroc(y_true, y_pred)
    aupr = compute_aupr(y_true, y_pred)
    return(acc, auroc, aupr)

# Function : custom_scoring_function_baseline
# Description : Custom scoring function (when there is hyperparameter tuning)
# Parameters :
# - estimator : the model
# - x : the training data
# - y : the test data
# Results : 
# - scoring_dict : dictionary containing the scoring metrics
# Note: this function is applied to both training and validation data sets
def custom_scoring_function_baseline(estimator, x, y):
    # Compute probabilities of each sequence belonging to either class
    class_probs = estimator.predict_proba(X = x)
    # Compute AUROCs of predictions
    acc, auc, aupr = evaluate(y, class_probs[:,1])
    scoring_dict = {"accuracy": acc, "auroc": auc, "aupr": aupr}
    return(scoring_dict)

# Function : custom_scoring_function_contexts
# Description : Custom scoring function (when there is hyperparameter tuning)
# Parameters :
# - estimator : the model
# - x : the training data
# - y : the test data
# Results : 
# - scoring_dict : dictionary containing the scoring metrics
# Note: this function is applied to both training and validation data sets
def custom_scoring_function_contexts(estimator, x, y):
    # Compute probabilities of each sequence belonging to either class
    class_probs = estimator.predict_proba(X = x)
    # Compute AUROCs of predictions
    acc, auc, aupr = evaluate(y, class_probs[:,1])
    scoring_dict = {"accuracy": acc, "auroc": auc, "aupr": aupr}
    return(scoring_dict)

# Function : parse_randomized_search_obj
# Description : Parse the CV results from the Randomized Search CV object to get the metric of interest
def parse_randomized_search_obj(search_obj, search_keys, metric, training_cv_groups):
    train_indices = list(np.where(np.array(["split" in key and "train_" + metric in key for key in search_keys]))[0])
    train_dict = {}
    for i in range(0, len(train_indices)):
        train_dict[i] = search_obj.cv_results_[search_keys[train_indices[i]]][0]
    # Get test performance results
    test_indices = list(np.where(np.array(["split" in key and "test_" + metric in key for key in search_keys]))[0])
    test_dict = {}
    for i in range(0, len(test_indices)):
        test_dict[i] = search_obj.cv_results_[search_keys[test_indices[i]]][0]
    # Format all performance results
    CV_results_df = pd.concat([pd.DataFrame([training_cv_groups]), pd.DataFrame([train_dict]), pd.DataFrame([test_dict])])
    CV_results_df = CV_results_df.transpose()
    CV_results_df.columns = ["CV_group", "train", "test"]
    CV_results_df = pd.concat([CV_results_df, pd.DataFrame({"CV_group": "avg", \
                                                            "train": [round(search_obj.cv_results_["mean_train_" + metric][0], 6)], \
                                                            "test": [round(search_obj.cv_results_["mean_test_" + metric][0], 6)]})])
    CV_results_df = pd.concat([CV_results_df, pd.DataFrame({"CV_group": "stdev", \
                                                            "train": [round(search_obj.cv_results_["std_train_" + metric][0], 6)], \
                                                            "test": [round(search_obj.cv_results_["std_test_" + metric][0], 6)]})])
    CV_results_df.set_index(CV_results_df.columns[0], inplace = True, drop = True)
    return(CV_results_df)

# Function : assess_CV
# Description : Extract CV results from the Randomized Search CV object
def assess_CV(search_obj, training_cv_groups):
    search_keys = list(search_obj.cv_results_.keys())
    accuracy_df = parse_randomized_search_obj(search_obj, search_keys, "accuracy", training_cv_groups)
    auroc_df = parse_randomized_search_obj(search_obj, search_keys, "auroc", training_cv_groups)
    aupr_df = parse_randomized_search_obj(search_obj, search_keys, "aupr", training_cv_groups)
    return(accuracy_df, auroc_df, aupr_df)
 



