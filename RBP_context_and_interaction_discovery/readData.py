# Shaimae Elhajjajy
# July 14, 2022
# Function to load in all data (used in multiple scripts)

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Import libraries
import pandas as pd
import pickle
from tensorflow import keras

# Function : load_data
# Description : Read in the data frame containing all training sequences
# Parameters : 
# - all_seq_file : the name of the file to read in
# Returns : 
# - all_seq_df : the Pandas data frame containing all training sequences
def load_seq(all_seq_file):
    all_seq_df = pd.read_csv(all_seq_file, sep =  "\t", header = None)
    # all_seq_df.columns = ["chr", "start", "end", "label", "1000", "strand", "log2_enrichment", "neg_log10_pvalue", "peakID", \
    #                         "peak_length", "peak_signal_mean", "peak_signal_std", "peak_signal_min", "peak_signal_max", \
    #                         "peak_signal_coverage", "seq", "seq_signal_mean", "seq_signal_std", "seq_signal_min", "seq_signal_max", \
    #                         "seq_signal_coverage",  "y_label", "CV_group"]
    all_seq_df.columns = ["chr", "start", "end", "label", "1000", "strand", "peakID", "region", "seq", \
                            "seq_signal_mean", "seq_signal_std", "seq_signal_min", "seq_signal_max", "seq_signal_coverage", \
                            "y_label", "CV_group"]

    return(all_seq_df)

# Function : load_model
# Description : Read in a pre-trained keras model
# Parameters : 
# - model_path : path to the saved keras model (Tensorflow SavedModel format)
# Returns :
# - model : the trained model
def load_model(model_path):
    model = keras.models.load_model(model_path)
    
    return(model)

# Function : load_object
# Description : Read in a saved object
# Parameters : 
# - obj_path : path to the saved object
# Returns : 
# - obj : the saved object
def load_object(obj_path):
    with open(obj_path, "rb") as filehandler:
        obj = pickle.load(filehandler)
    filehandler.close()
    return(obj)



