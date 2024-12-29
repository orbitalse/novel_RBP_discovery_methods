# Shaimae Elhajjajy
# July 13, 2022
# Define Classifier class for import to other scripts

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Import libraries
import encoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import subprocess
from tensorflow import keras

class Classifier:
    def __init__(self, hidden_shape = 256, out_shape = 1, in_shape = 1024 + 1, \
                    optimizer = "adam", loss_function = "binary_crossentropy", \
                    num_epochs = 10, embedding_dim = 8, dropout_rate = 0.5, \
                    num_filters = 32, kernel_length = 4, kernel_stride = 1, pool_size = 2, \
                    batch_size = 32):
        self.model = keras.models.Sequential()
        self.in_shape = in_shape
        self.hidden_shape = hidden_shape
        self.out_shape = out_shape
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = num_epochs
        self.tokenizer = None
        self.fitted = False
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        #self.dropout = False
        self.dropout=True # Edit made 04/11/24 to test reproducibility of motif from different MIL runs
        self.num_filters = num_filters # number of convolution filters
        self.kernel_length = kernel_length # length of 1D convolutional window; value = n-grams
        self.kernel_stride = kernel_stride # step size of filter as it moves down the embedding
        self.pool_size = pool_size # size of the pooling window; number of cells from which the maximum will be taken
        self.batch_size = batch_size
    
    # Tokenize into Bag-of-Words
    def tokenize_old(self, training_strings, test_strings, mode):
        self.tokenizer = encoder.Tokenizer()
        self.tokenizer.fit(training_strings)
        X_train = self.encode_old(training_strings, mode)
        X_test = self.encode_old(test_strings, mode)

        return(X_train, X_test)
    
    # Tokenize sequences into kmers and develop the dictionary
    def tokenize(self, training_strings):
        self.tokenizer = encoder.Tokenizer()
        self.tokenizer.fit(training_strings)

        return(self.tokenizer)
    
    def encode_matrix(self, strings, mode):
        X = self.tokenizer.encode_matrix(strings, mode)
        return(X)

    def encode_integers_baseline(self, seq_strings):
        X, max_length = self.tokenizer.encode_integers_baseline(seq_strings)
        return(X, max_length)

    def encode_integers_contexts(self, ctxt_strings, num_ctxt, num_ctxt_per_seq):
        X, max_length = self.tokenizer.encode_integers_contexts(ctxt_strings, num_ctxt, num_ctxt_per_seq)
        return(X, max_length)
    
    def embed_only(self, vocab_size, max_length):
        self.model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, input_length = max_length))
        self.model.compile(loss = "mse", \
                            optimizer = "rmsprop") # MSE and RMSprop used in keras Embedding Layer API
        self.model.summary()
        return(self)
    
    def build_BoW(self):
        self.model.add(keras.layers.Dense(units = self.hidden_shape, \
                                            input_shape = (self.in_shape,), \
                                            activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
        if (self.dropout == True):
            self.model.add(keras.layers.Dropout(rate = self.dropout_rate))
        self.model.add(keras.layers.Dense(units = self.out_shape, \
                                        activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
        self.model.compile(loss = self.loss_function, \
                            optimizer = self.optimizer, \
                            metrics = ["accuracy", "AUC"]) # loss function for binary classification
        self.model.summary()
        return(self)

    def build_w_Embedding(self, vocab_size, max_length):
        self.model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, input_length = max_length))
        self.model.add(keras.layers.Flatten()) # flatten to a 1D vector for input to a Dense layer
        self.model.add(keras.layers.Dense(units = self.hidden_shape, \
                                            activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
        if (self.dropout == True):
            self.model.add(keras.layers.Dropout(rate = self.dropout_rate))
        self.model.add(keras.layers.Dense(units = self.out_shape, \
                                        activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
        self.model.compile(loss = self.loss_function, \
                            optimizer = self.optimizer, \
                            metrics = ["accuracy", "AUC"]) # loss function for binary classification
        self.model.summary()
        return(self)
    
    def build_wo_Embedding(self, max_length):
        self.model.add(keras.layers.InputLayer(input_shape = (max_length, self.embedding_dim,)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units = self.hidden_shape, \
                                            activation = "relu")) # Single dense layer w/ 256 neurons & ReLU activation
        if (self.dropout == True):
            self.model.add(keras.layers.Dropout(rate = self.dropout_rate))
        self.model.add(keras.layers.Dense(units = self.out_shape, \
                                            activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
        self.model.compile(loss = self.loss_function, \
                            optimizer = self.optimizer, \
                            metrics = ["accuracy", "AUC"]) # loss function for binary classification
        self.model.summary()
        return(self)
    
    def build_w_CNN(self, vocab_size, max_length):
        self.model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, input_length = max_length))
        self.model.add(keras.layers.Conv1D(filters = self.num_filters, \
                                            kernel_size = self.kernel_length, \
                                            strides = self.kernel_stride, \
                                            activation = "relu"))
        self.model.add(keras.layers.MaxPooling1D(pool_size = self.pool_size))
        self.model.add(keras.layers.Flatten()) # flatten to 1D vector for input to a Dense layer
        self.model.add(keras.layers.Dense(units = self.hidden_shape, \
                                            activation = "relu"))
        if (self.dropout == True):
            self.model.add(keras.layers.Dropout(rate = self.dropout_rate))
        self.model.add(keras.layers.Dense(units = self.out_shape, \
                                            activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
        self.model.compile(loss = self.loss_function, \
                            optimizer = self.optimizer, \
                            metrics = ["accuracy", "AUC"]) # loss function for binary classification
        self.model.summary()
        return(self)

    def build_w_CNN_LSTM(self, vocab_size, max_length):
        self.model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, input_length = max_length))
        self.model.add(keras.layers.Conv1D(filters = self.num_filters, \
                                            kernel_size = self.kernel_length, \
                                            strides = self.kernel_stride, \
                                            activation = "relu"))
        self.model.add(keras.layers.MaxPooling1D(pool_size = self.pool_size))
        self.model.add(keras.layers.LSTM(units = 64, dropout = 0.1, recurrent_dropout = 0.5))
        self.model.add(keras.layers.Dense(units = self.hidden_shape, \
                                            activation = "relu"))
        if (self.dropout == True):
            self.model.add(keras.layers.Dropout(rate = self.dropout_rate))
        self.model.add(keras.layers.Dense(units = self.out_shape, \
                                            activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
        self.model.compile(loss = self.loss_function, \
                            optimizer = self.optimizer, \
                            metrics = ["accuracy", "AUC"]) # loss function for binary classification
        self.model.summary()
        return(self)
    
    def build_w_CNN_biLSTM(self, vocab_size, max_length):
        self.model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, input_length = max_length))
        self.model.add(keras.layers.Conv1D(filters = self.num_filters, \
                                            kernel_size = self.kernel_length, \
                                            strides = self.kernel_stride, \
                                            activation = "relu"))
        self.model.add(keras.layers.MaxPooling1D(pool_size = self.pool_size))
        self.model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = 64, dropout = 0.1, recurrent_dropout = 0.5)))
        self.model.add(keras.layers.Dense(units = self.hidden_shape, \
                                            activation = "relu"))
        if (self.dropout == True):
            self.model.add(keras.layers.Dropout(rate = self.dropout_rate))
        self.model.add(keras.layers.Dense(units = self.out_shape, \
                                            activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
        self.model.compile(loss = self.loss_function, \
                            optimizer = self.optimizer, \
                            metrics = ["accuracy", "AUC"]) # loss function for binary classification
        self.model.summary()
        return(self)
    
    def build_w_CNN_GRU(self, vocab_size, max_length):
        self.model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = self.embedding_dim, input_length = max_length))
        self.model.add(keras.layers.Conv1D(filters = self.num_filters, \
                                            kernel_size = self.kernel_length, \
                                            strides = self.kernel_stride, \
                                            activation = "relu"))
        self.model.add(keras.layers.MaxPooling1D(pool_size = self.pool_size))
        self.model.add(keras.layers.GRU(units = 64, dropout = 0.1, recurrent_dropout = 0.5))
        self.model.add(keras.layers.Dense(units = self.hidden_shape, \
                                            activation = "relu"))
        if (self.dropout == True):
            self.model.add(keras.layers.Dropout(rate = self.dropout_rate))
        self.model.add(keras.layers.Dense(units = self.out_shape, \
                                            activation = "sigmoid")) # Sigmoid for binary classification (outputs probability btwn 0 and 1)
        self.model.compile(loss = self.loss_function, \
                            optimizer = self.optimizer, \
                            metrics = ["accuracy", "AUC"]) # loss function for binary classification
        self.model.summary()
        return(self)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs = self.epochs, batch_size = self.batch_size)
        self.fitted = True
        return(self)

    def refit(self, X_train, y_train):
        self.fitted = False
        self.fit(X_train, y_train)
        return(self)

    def predict(self, X_test):
        predictions = self.model.predict(X_test, verbose = 0)
        return(predictions)

    def evaluate(self, X_test, y_test):
        loss, acc, auc = self.model.evaluate(X_test, y_test, verbose = 0)
        return(loss, acc, auc)
    
    def compute_acc(self, y_true, y_pred):
        acc_obj = keras.metrics.BinaryAccuracy()
        acc_obj.update_state(y_true, y_pred)
        acc = acc_obj.result().numpy()
        return(acc)

    def compute_auroc(self, y_true, y_pred):
        roc_curve = keras.metrics.AUC(curve = "ROC")
        roc_curve.update_state(y_true, y_pred)
        auc = roc_curve.result().numpy()
        return(auc, roc_curve)

    def compute_aupr(self, y_true, y_pred):
        pr_curve = keras.metrics.AUC(curve = "PR")
        pr_curve.update_state(y_true, y_pred)
        aupr = pr_curve.result().numpy()
        return(aupr, pr_curve)
       
    def get_contingency_table_metrics(self, metric_obj, out_file):
        thresholds = metric_obj.thresholds
        tp = list(np.array(metric_obj.true_positives))
        fp = list(np.array(metric_obj.false_positives))
        tn = list(np.array(metric_obj.true_negatives))
        fn = list(np.array(metric_obj.false_negatives))
        tpr = [(tp[i] / (tp[i] + fn[i])) for i in range(0, len(tp))]
        fpr = [(fp[i] / (fp[i] + tn[i])) for i in range(0, len(fp))]
        tnr = [(tn[i] / (tn[i] + fp[i])) for i in range(0, len(tn))]
        fnr = [(fn[i] / (fn[i] + tp[i])) for i in range(0, len(fn))]
        metrics_df = pd.DataFrame({"thresholds": thresholds, "tp": tp, "fp": fp, "tn": tn, "fn": fn, \
                                    "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr})
        metrics_df.to_csv(out_file, sep = "\t", index = False)
        return(metrics_df)

    # Updated evaluation function to include AUPR computation
    def evaluate_plus(self, X_test, y_test, y_pred, out_file):
        loss, acc, auc = self.model.evaluate(X_test, y_test, verbose = 0)
        acc = self.compute_acc(y_test, y_pred)
        auroc, roc_curve = self.compute_auroc(y_test, y_pred)
        aupr, pr_curve = self.compute_aupr(y_test, y_pred)
        auroc_contingency = self.get_contingency_table_metrics(roc_curve, out_file + ".ROC_curve.tsv")
        aupr_contingency = self.get_contingency_table_metrics(pr_curve, out_file + ".PR_curve.tsv")
        return(loss, acc, auroc, aupr)

    # Use max of instance predictions to evaluate bag performance 
    def aggregate_max(self, instance_pred, threshold):
        bag_pred = []
        seq_ids = instance_pred.seq_id.unique()
        for i in range(0, len(seq_ids)):
            seq_contexts = instance_pred[instance_pred.seq_id == seq_ids[i]]
            max_pred = seq_contexts.predicted_y.max()
            bag_pred.append(max_pred)
        return(bag_pred)

    # Use average of instance predictions to evaluate bag performance
    def aggregate_avg(self, instance_pred, threshold):
        bag_pred = []
        seq_ids = instance_pred.seq_id.unique()
        for i in range(0, len(seq_ids)):
            seq_contexts = instance_pred[instance_pred.seq_id == seq_ids[i]]
            avg_pred = seq_contexts.predicted_y.mean()
            bag_pred.append(avg_pred)
        return(bag_pred)

    # Use majority vote of instance predictions to evaluate bag performance
    def aggregate_vote(self, instance_pred, threshold):
        bag_pred = []
        seq_ids = instance_pred.seq_id.unique()
        for i in range(0, len(seq_ids)):
            seq_contexts = instance_pred[instance_pred.seq_id == seq_ids[i]]
            num_above = seq_contexts[seq_contexts.predicted_y > threshold].shape[0]
            num_below = seq_contexts[seq_contexts.predicted_y <= threshold].shape[0]
            if (num_above > num_below):
                bag_pred.append(1)
            else:
                bag_pred.append(0)
        return(bag_pred)
    
    # Use threshold for number of instances that must be positive to evaluate bag performance
    def aggregate_soft(self, instance_pred, pred_threshold, instance_threshold):
        bag_pred = []
        seq_ids = instance_pred.seq_id.unique()
        for i in range(0, len(seq_ids)):
            seq_contexts = instance_pred[instance_pred.seq_id == seq_ids[i]]
            num_above = seq_contexts[seq_contexts.predicted_y > pred_threshold].shape[0]
            if (num_above > instance_threshold):
                bag_pred.append(1)
            else:
                bag_pred.append(0)
        return(bag_pred)

    # Function to evaluate bag performance from instance predictions
    def evaluate_bag(self, bag_labels, instance_pred, threshold, aggregation_method, out_file):
        if (aggregation_method == "max"):
            bag_pred = self.aggregate_max(instance_pred, threshold)
        elif (aggregation_method == "avg"):
            bag_pred = self.aggregate_avg(instance_pred, threshold)
        elif (aggregation_method == "vote"):
            bag_pred = self.aggregate_vote(instance_pred, threshold)
        # elif (aggregation_method == "soft"):
        #     thresholds = [1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,97]
        #     acc_list = []
        #     auroc_list = []
        #     aupr_list = []
        #     for t in thresholds:
        #         bag_pred = self.aggregate_soft(instance_pred, threshold, t)
        #         acc = self.compute_acc(bag_labels, bag_pred)
        #         auroc = self.compute_auroc(bag_labels, bag_pred)
        #         aupr = self.compute_aupr(bag_labels, bag_pred)
        #         acc_list.append(acc)
        #         auroc_list.append(auroc)
        #         aupr_list.append(aupr)
        #         tp, tn, fp, fn, tpr, tnr, fpr, fnr = self.get_contingency_table_metrics(bag_labels, bag_pred)
        #         performance_df = pd.DataFrame({"acc": [acc], "auroc": [auroc], "aupr": [aupr], \
        #                                 "tp": [tp], "tn": [tn], "fp": [fp], "fn": [fn], \
        #                                 "tpr": [tpr], "tnr": [tnr], "fpr": [fpr], "fnr": [fnr]})
        #         performance_df.to_csv(out_file + "." + str(t) + ".tsv", index = False)
        #     # Plot accuracy as a function of instance threshold
        #     plt.plot(thresholds, acc_list, label = "accuracy")
        #     plt.xlabel("Thresholds for # instances") 
        #     plt.ylabel("Accuracy")
        #     plt.ylim([0, 1])
        #     # Plot AUROC as a function of instance threshold
        #     plt.plot(thresholds, auroc_list, label = "AUROC")
        #     plt.xlabel("Thresholds for # instances") 
        #     plt.ylabel("AUROC")
        #     plt.ylim([0, 1])
        #     # Plot AUPR as a function of instance threshold
        #     plt.plot(thresholds, aupr_list, label = "AUPR")
        #     plt.legend(loc = "lower right")
        #     plt.xlabel("Thresholds for # instances") 
        #     plt.ylabel("AUPR")
        #     plt.ylim([0, 1])
        #     plt.savefig(out_file + ".png")
        #     plt.close()
        #     return(acc_list, auroc_list, aupr_list)
        acc = self.compute_acc(bag_labels, bag_pred)
        auroc, roc_curve = self.compute_auroc(bag_labels, bag_pred)
        aupr, pr_curve = self.compute_aupr(bag_labels, bag_pred)
        self.get_contingency_table_metrics(roc_curve, out_file + ".ROC_curve.tsv")
        self.get_contingency_table_metrics(pr_curve, out_file + ".PR_curve.tsv")
        return(acc, auroc, aupr)
    
    def save_model(self, filename):
        self.model.save(filename)
        return()
    
    def save_object(self, filename):
        filehandler = open(filename, "wb")
        pickle.dump(self, filehandler)
        filehandler.close()
        return()

    def plot_curves_HP(self, model_type):
        subprocess.call(["Rscript", "plot_final_curves.HP.R", model_type])
        return()
    
    def plot_curves_baseline_on_contexts(self, model_type):
        subprocess.call(["Rscript", "plot_final_curves.baseline_on_contexts.R", model_type])
        return()
    
    def plot_curves_contexts_relabeled(self, model_type, iteration):
        subprocess.call(["Rscript", "plot_final_curves.contexts_relabeled.R", model_type, str(iteration)])
        return()
    
    # Save the classifier object
    def save_object(self, filename):
        filehandler = open(filename, "wb")
        pickle.dump(self, filehandler)
        filehandler.close()
        return()




