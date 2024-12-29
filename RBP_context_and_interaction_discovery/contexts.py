# Shaimae Elhajjajy
# July 13, 2022
# Define kmer, context, and sequence classes for import to other scripts

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Class : Kmer
# Description : class containing a kmer and its length; kmer may be target or part of context

class Kmer:
    # Initialize Kmer object with the kmer and its length
    def __init__(self, kmer = None):
        if kmer:
            self.kmer = kmer # kmer 
            self.length = len(kmer) # kmer length
        else:
            self.kmer = ""
            self.length = 0

# Class : KmerContext
# Description : class containing all information related to contexts
class KmerContext:
    # Initialize KmerContext object with left and right context of a kmer
    def __init__(self, target, position, label, left_context = None, right_context = None):
        self.target = target
        self.position = position
        self.seq_label = label # inherited label from the parent sequence
        self.left_context = left_context # context to left of kmer
        self.right_context = right_context # context to right of kmer
        self.context_as_string = ""
        self.id = ""
        self.original_seq = None
        self.chr = None
        self.start_coord = None
        self.end_coord = None

        # SSL label
        self.ssl_label = 0 # Start as 0, update if this context contains the most enriched kmer

# Class : Sequence
# Description : class containing all information related to sequences
class Sequence:

    # Initialize Sequence object with full RNA sequence and its properties
    def __init__(self, seq, id, label, cv_group, chr, start_coord, end_coord, strand):
        self.seq = seq # sequence
        self.id = id # sequence number (in the form of SEQ#, where # is an integer)
        self.length = len(seq) # sequence length
        self.label = label # 1 for bound, 0 for unbound
        self.cv_group = cv_group # chromCV group
        self.chr = chr
        if (start_coord == "."):
            self.start_coord = None
            self.end_coord = None
        else:
            self.start_coord = int(start_coord)
            self.end_coord = int(end_coord)
        self.strand = strand

        # For baseline
        self.split_seq = [] # A list of kmers (as Kmer objects) in the split sequence
        self.split_seq_length = "" # Number of kmers in the split sequence
        self.split_seq_as_strings = [] # String containing kmers in the sequence separated by a space

        # For contexts
        self.contexts = [] # A list of dictionaries of the format {Kmer, position, KmerContext, label}
        self.contexts_as_strings = [] # A list of strings of the format "left_context kmer right_context"
        self.context_labels = [] # A list of length len(self.contexts) containing self.label
        self.context_seq_ids = [] # A list of length len(self.contexts) containing self.id

        # For baseline
        self.baseline_ksize_list = []
        self.baseline_step_list = [] 

        self.baseline_seq_strings = []
        self.baseline_seq_objects = []

        self.context_strings = []
        self.context_objects = []
        self.context_ids = []

        # For SSL approach
        self.most_enriched_kmer = ""
        self.most_enriched_kmer_position = 0
    
    # Function : split_seq
    # Description : split each sequence into kmers (analogous to splitting sentences into words)
    # Parameters :
    # - k : the size of the kmer
    # - step : the step size of the sliding window
    # Returns :
    # - split_seq : a list containing the kmers (as Kmer objects) in the split sequence
    # --- OLD CODE
    def split_sequence(self, k, step):
        self.k = k
        self.step = step

        split_seq = []
        for i in range(0, (len(self.seq) - k + 1), step):
            kmer = Kmer(self.seq[i:(i + k)])
            split_seq.append(kmer)
        
        self.split_seq = split_seq
        self.split_seq_length = len(split_seq)

        return(self.split_seq)

    # Function : convert_split_seq_to_string
    # Description : Combine the list of strings containing kmers in the sequence to a single string
    #               with each kmer separated by a space. 
    # Parameters : 
    # - NA
    # Returns : 
    # self.split_seq_as_strings : a string containing the kmers in the sequence separated by a space
    # --- OLD CODE
    def convert_split_seq_to_string(self):
        split_seq_as_string = []
        for i in range(len(self.split_seq)):
            split_seq_as_string.append(self.split_seq[i].kmer)
        self.split_seq_as_strings = ' '.join(split_seq_as_string)

        return(self.split_seq_as_strings)

    # Function : get_right_context
    # Description : for a given kmer, extract its context to the right
    # Parameters :
    # - position : the position of the kmer in the split sequence (must be specified in the case of duplicates)
    # - num_context_kmers : the number of context kmers to extract
    # - gap_size : the gap size between context kmers
    # Returns : 
    # - right_context : a list containing the right context of the kmer (as Kmer objects)
    # --- OLD CODE
    def get_right_context(self, position, num_context_kmers, gap_size):
        right_context = []
        for i in range(0, num_context_kmers):
            context_position = position + ((i + 1) * (gap_size)) + (i + 1)
            if (context_position >= len(self.split_seq)):
                context = Kmer()
            else:
                context = self.split_seq[context_position]
            right_context.append(context)
        return(right_context)

    # Function : get_left_context
    # Description : for a given kmer, extract its context to the left
    # Parameters :
    # - position : the position of the kmer in the split sequence (must be specified in the case of duplicates)
    # - num_context_kmers : the number of context kmers to extract
    # - gap_size : the gap size between context kmers
    # Returns : 
    # - left_context : a list containing the left context of the kmer (as Kmer objects)
    # --- OLD CODE
    def get_left_context(self, position, num_context_kmers, gap_size):
        left_context = []
        for i in range(0, num_context_kmers):
            context_position = position - ((i + 1) * (gap_size)) - (i + 1)
            if (context_position < 0):
                context = Kmer()
            else:
                context = self.split_seq[context_position]
            left_context.append(context)
        left_context.reverse()
        return(left_context)

    # Function : map_context_to_seq
    # Description : for a given context, collapse [left_context target right_context] to recover the sub-sequence from the original sequence
    # Parameters :
    # - context : the KmerContext object to map back to the sequence
    # - num_context_kmers : the number of context kmers on each side of the target (central) kmer
    # - gap_size : the gap size between context kmers
    # Returns :
    # - original_seq : the original sequence that the context maps back to
    # --- OLD CODE
    def map_context_to_seq(self, context, k, step, num_context_kmers, gap_size):
        leftmost_context_position = context.position - num_context_kmers
        rightmost_context_position = context.position + num_context_kmers
        max_contexts = self.split_seq_length
        if (leftmost_context_position < 0 ):
            leftmost_context_position = 0
        if (rightmost_context_position > max_contexts):
            rightmost_context_position = max_contexts
        pointer = 0
        for i in range(0, rightmost_context_position):
            if (i == leftmost_context_position):
                context_start = pointer
            if (i == rightmost_context_position):
                context_end = pointer + k
                break
            pointer += step
        context.original_seq = self.seq[context_start:context_end]

        context.chr = self.chr
        context.strand = self.strand
        if (context.strand == "+"):
                context.start_coord = self.start_coord + context_start
                context.end_coord = self.start_coord + context_end
        elif (context.strand == "-"):
            context.start_coord = self.end_coord - context_end
            context.end_coord = self.end_coord - context_start
        return(context)

    # Function : get_kmer_context
    # Description : for a given kmer, extract its left and right context
    # Parameters : 
    # - kmer : the kmer for which the context is being extracted
    # - position : the position of the kmer in the split sequence
    # - num_context_kmers : the number of context kmers to extract
    # - gap_size : the gap size between context kmers
    # Returns : 
    # - kmer_context : a KmerContext object containing (1) the kmer, as a Kmer object,
    #                  (2) the left context, as a list of Kmer objects, and (3) the right context, as a list of Kmer objects
    # --- OLD CODE
    def get_kmer_context(self, target, position, k, step, num_context_kmers, gap_size):
        right_context = self.get_right_context(position, num_context_kmers, gap_size)
        left_context = self.get_left_context(position, num_context_kmers, gap_size)
        target_context = KmerContext(target, left_context, right_context)

        target_context.position = position
        target_context = self.map_context_to_seq(target_context, k, step, num_context_kmers, gap_size)
        
        return(target_context)
    
    # Function : get_seq_contexts
    # Description : for a given sequence, extract all contexts for every kmer in that sequence
    # Parameters : 
    # - k : the size of the kmer
    # - step : the step size of the sliding window
    # - num_context_kmers : the number of context kmers to extract
    # - gap_size : the gap size between context kmers
    # Returns : 
    # - self.contexts : a list of dictionaries (one for every kmer in the sequence) of the form 
    #                   {Kmer, position, KmerContext}
    # --- OLD CODE
    def get_seq_contexts(self, k, step, num_context_kmers, gap_size):
        if (len(self.split_seq) == 0):
            self.split_sequence(k, step)
        for i in range(0, len(self.split_seq)):
            target = self.split_seq[i]
            target_context = self.get_kmer_context(target, i, k, step, num_context_kmers, gap_size)
            self.contexts.append(target_context)

        return(self.contexts)

    # Function : extract_contexts
    # Description : extract the left and right contexts as lists of strings (rather than lists of Kmer objects)
    # Parameters : 
    # - context : a single context dictionary of the form {Kmer, position, KmerContext}
    # --- OLD CODE
    def extract_contexts(self, context):
        left_context = []
        right_context = []
        for l in context.left_context:
            left_context.append(l.kmer)
        for r in context.right_context:
            right_context.append(r.kmer)
        return(left_context, right_context)
           
    # Function : convert_seq_contexts_to_string
    # Description : Convert the list of dictionaries in the form {Kmer, position, KmerContext} into a list of strings in 
    #               the form "left_context target right_context"
    # Parameters : 
    # - NA
    # Returns : 
    # - context_strings : the contexts dictionaries reformatted as a list of "left_context target right_context", where each element is separated by a space
    # --- OLD CODE
    def convert_seq_contexts_to_string(self):
        contexts_as_strings = []
        for context in self.contexts:
            left_context, right_context = self.extract_contexts(context)
            left_context = ' '.join(left_context)
            right_context = ' '.join(right_context)
            context_string = left_context + ' ' + context.target.kmer + ' ' + right_context
            contexts_as_strings.append(context_string)
        self.contexts_as_strings = contexts_as_strings

        return(contexts_as_strings)
    
    # Function : get_seq_context_labels
    # Description : Set the y label (bound or unbound) for every context in the sequence
    # Parameters : 
    # - NA
    # Returns : 
    # - self.context_labels : A list of length len(self.contexts) containing self.label
    # --- OLD CODE
    def get_seq_context_labels(self):
        self.context_labels = [self.label] * len(self.contexts)

        return(self.context_labels)

    # Function : get_seq_context_ids(self)
    # Description : Set the id for every context in the sequence
    # Parameters :
    # - NA
    # Returns :
    # - self.context_seq_ids : A list of length len(self.contexts) containing self.id
    # --- OLD CODE
    def get_seq_context_ids(self):
        self.context_seq_ids = [self.id] * len(self.contexts)

        return(self.context_seq_ids)

    # -------------------------------------------- NEW CODE --------------------------------------------

    def get_contexts(self, target_size, target_step, kmer_size, kmer_step, num_ctxt):

        # Initialize target positions
        target_start = 0
        target_end = target_start + target_size
        # Initialize context id (number in sequence)
        ctxt_id = 1

        # Make sure targets don't go too far along the sequence
        while ((target_start + target_size) <= self.length):
            # Get target
            target_kmer = self.seq[target_start:target_end]
            position = target_start
            # Get size of context on each side
            half_ctxt_size = ((num_ctxt * kmer_size) - ((num_ctxt - 1) * (kmer_size - kmer_step)))
            # Get left contexts if conditions are met
            left_ctxt_list = []
            if (target_start - half_ctxt_size >= 0): # Enough space for full left context
                left_ctxt_kmer_start = target_start - half_ctxt_size
                left_ctxt_kmer_end = left_ctxt_kmer_start + kmer_size
                ctxt_start = left_ctxt_kmer_start # Save position of left-most context
                for n in range(0, num_ctxt):
                    left_ctxt_kmer = self.seq[left_ctxt_kmer_start:left_ctxt_kmer_end]
                    left_ctxt_list.append(left_ctxt_kmer)
                    left_ctxt_kmer_start += kmer_step
                    left_ctxt_kmer_end += kmer_step
            # If condition not met, solve for number of contexts that can fit to left of target
            elif (target_start - half_ctxt_size < 0):
                num_ctxt_possible = int((target_start - kmer_size + kmer_step) / kmer_step) # sub ctxt_size for distance from sequence start to target start
                if (num_ctxt_possible > 0):
                    num_blanks = num_ctxt - num_ctxt_possible
                    for i in range(0, num_blanks):
                        left_ctxt_list.append("")
                    # compute start of left context based on number of contexts that fit to the left of the target
                    left_ctxt_kmer_start = target_start - ((num_ctxt_possible * kmer_size) - ((num_ctxt_possible - 1) * (kmer_size - kmer_step))) 
                    left_ctxt_kmer_end = left_ctxt_kmer_start + kmer_size
                    ctxt_start = left_ctxt_kmer_start # Save position of left-most context
                    for n in range(0, num_ctxt_possible):
                        left_ctxt_kmer = self.seq[left_ctxt_kmer_start:left_ctxt_kmer_end]
                        left_ctxt_list.append(left_ctxt_kmer)
                        left_ctxt_kmer_start += kmer_step
                        left_ctxt_kmer_end += kmer_step
                else:
                    for i in range(0, num_ctxt):
                        left_ctxt_list.append("")
                    ctxt_start = target_start
            # Get right contexts if conditions are met
            right_ctxt_list = []
            if (target_end + half_ctxt_size <= self.length): # Enough space for full right context
                right_ctxt_kmer_start = target_end
                right_ctxt_kmer_end = right_ctxt_kmer_start + kmer_size
                for n in range(0, num_ctxt):
                    right_ctxt_kmer = self.seq[right_ctxt_kmer_start:right_ctxt_kmer_end]
                    right_ctxt_list.append(right_ctxt_kmer)
                    right_ctxt_kmer_start += kmer_step
                    right_ctxt_kmer_end += kmer_step
                ctxt_end = right_ctxt_kmer_end - kmer_step # Save position of right-most context
            # If condition not met, solve for number of contexts that can fit to right of target
            elif (target_end + half_ctxt_size > self.length):
                num_ctxt_possible = int(((self.length - target_end) - kmer_size + kmer_step) / kmer_step) # sub ctxt_size for distance from target end to sequence end 
                if (num_ctxt_possible > 0):
                    num_blanks = num_ctxt - num_ctxt_possible
                    for i in range(0, num_blanks):
                        right_ctxt_list.append("")
                    right_ctxt_kmer_start = target_end
                    right_ctxt_kmer_end = right_ctxt_kmer_start + kmer_size
                    for n in range(0, num_ctxt_possible):
                        right_ctxt_kmer = self.seq[right_ctxt_kmer_start:right_ctxt_kmer_end]
                        right_ctxt_list.append(right_ctxt_kmer)
                        right_ctxt_kmer_start += kmer_step
                        right_ctxt_kmer_end += kmer_step
                    ctxt_end = right_ctxt_kmer_end - kmer_step # Save position of right-most context
                else:
                    for i in range(0, num_ctxt):
                        right_ctxt_list.append("")
                    ctxt_end = target_end

            # Format as string
            if (right_ctxt_list.count("") != len(right_ctxt_list)):
                while right_ctxt_list[0] == "":
                    right_ctxt_list.append(right_ctxt_list.pop(right_ctxt_list.index("")))
            left_ctxt_string = " ".join(left_ctxt_list)
            right_ctxt_string = " ".join(right_ctxt_list)
            if (left_ctxt_string == ""):
                ctxt_string = target_kmer + " " + right_ctxt_string
            elif (right_ctxt_string == ""):
                ctxt_string = left_ctxt_string + " " + target_kmer
            else:
                ctxt_string = left_ctxt_string + " " + target_kmer + " " + right_ctxt_string
            self.context_strings.append(ctxt_string)
            # Format as object
            left_ctxt_obj = [Kmer(i) for i in left_ctxt_list]
            right_ctxt_obj = [Kmer(i) for i in right_ctxt_list]
            ctxt_obj = KmerContext(Kmer(target_kmer), position, self.label, left_ctxt_obj, right_ctxt_obj)
            ctxt_obj.id = "ctxt_" + str(ctxt_id)
            self.context_ids.append(ctxt_obj.id)
            # Get original sequence
            ctxt_obj.original_seq = self.seq[ctxt_start:ctxt_end]
            # Add other attributes
            ctxt_obj.context_as_string = ctxt_string
            ctxt_obj.chr = self.chr
            ctxt_obj.strand = self.strand
            if (ctxt_obj.strand == "+"):
                ctxt_obj.start_coord = self.start_coord + ctxt_start
                ctxt_obj.end_coord = self.start_coord + ctxt_end
            elif (ctxt_obj.strand == "-"):
                ctxt_obj.start_coord = self.end_coord - ctxt_end
                ctxt_obj.end_coord = self.end_coord - ctxt_start
            self.context_objects.append(ctxt_obj)
            target_start += target_step
            target_end += target_step
            ctxt_id += 1

    def split_seq_into_kmers(self, k, step):
        split_seq = []
        for i in range(0, (len(self.seq) - k + 1), step):
            kmer = self.seq[i:(i + k)]
            split_seq.append(kmer)
        return(split_seq)

    def perform_multivariate_split(self, kmer_size_list, step_size_list):
        self.baseline_ksize_list = kmer_size_list
        self.baseline_step_list = step_size_list
        for i in range(0, len(kmer_size_list)):
            k = kmer_size_list[i]
            step = step_size_list[i]
            # Split sequence into list of kmers
            baseline_seq_list = self.split_seq_into_kmers(k, step)
            # Save baseline sequence as list of strings
            self.baseline_seq_strings.append(" ".join(baseline_seq_list))
            # Convert strings to Kmer objects
            self.baseline_seq_objects.append([Kmer(i) for i in baseline_seq_list])




