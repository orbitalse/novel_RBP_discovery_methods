#!/usr/bin/env python3

# Shaimae Elhajjajy
# December 20, 2023
# Fetch metadata eCLIP data and download peaks from the ENCODE portal using the REST API.

# Copyright (c) 2024 Shaimae I. Elhajjajy
# This code is licensed under the MIT License (see LICENSE.txt for details)

# Load packages
import json
import numpy as np
import os
import pandas as pd
import requests
import sys

# Read from command line
if (len(sys.argv) != 2):
    print("Enter the desired cell type {HepG2 | K562}.")
    sys.exit()

cell_type = sys.argv[1]

# ------------------------------------------------ FUNCTIONS ------------------------------------------------

def download_data(expID, accession, target, cell_type):
    url = "https://www.encodeproject.org/files/" + accession + "/@@download/" + accession + ".bed.gz"
    out_file = "./data/" + cell_type + "/" + target + "." + cell_type + "." + expID + "." + accession + ".bed.gz"
    content = requests.get(url).content
    f = open(out_file, "wb")
    f.write(content)
    f.close()

def fetch_data(expID):
    exp_url = "https://www.encodeproject.org/experiments/" + expID + "/?format=json"
    # Extract the experiment's metadata
    exp_response = requests.get(exp_url, headers = headers)
    exp_metadata = exp_response.json()
    # Loop through the experiment's files to find the desired ones (e.g., fastq, bam, tsv, etc.)
    for file_metadata in exp_metadata["files"]:
        # Filter based on your requirements
        # Note: there are many fields you can access with experiment information: biosample, replicates, file formats, output type, lab, etc.)
            if (file_metadata["output_type"] == "peaks") and \
                (file_metadata["file_format"] == "bed") and \
                    (file_metadata["assembly"] == "GRCh38") and \
                        (len(file_metadata["biological_replicates"]) == 2): # Get peaks from merged replicates
                        # ("preferred_default" in file_metadata): # Get replicate marked as preferred default by ENCODE
                # Get desired metadata for this file
                accession = file_metadata["accession"]
                target = exp_metadata["target"]["genes"][0]["symbol"]
                replicate = file_metadata["biological_replicates"]
                file_format = file_metadata["file_format"]
                output_type = file_metadata["output_type"]
                biosample = exp_metadata["biosample_ontology"]["term_name"]
                print(expID, "\t", accession, "\t", target, "\t", biosample, "\t", file_format, "\t", output_type, "\t", replicate)
                download_data(expID, accession, target, cell_type)

# -------------------------------------------------- MAIN --------------------------------------------------

if __name__ == "__main__":

    # Get URL from the ENCODE portal after filtering experiments with faceted browsing
    # This URL contains information about the list of experiments you are intersted in
    url = "https://www.encodeproject.org/search/?type=Experiment&status=released&internal_tags=ENCORE&assay_title=eCLIP" # Select eCLIP data
    url += "&biosample_ontology.term_name=" + cell_type # Filter by desired cell type
    url += "&limit=all" # List all experiments meeting criteria
    url += "&format=json" # Return metadata in JSON format

    headers = {'accept':'application/json'}

    # Request metadata from REST
    response = requests.get(url, headers = headers)

    # View metadata in JSON format (similar to a Python dictionary)
    metadata = response.json()
    
    # This line allows you to print metadata to the command line, so you can view the fields and determine which you want to access.
    # It can also be used to view experiment and file metadata (below)
    #print(json.dumps(metadata, indent = 4))

    # Create output directory
    if not os.path.exists("./data/" + cell_type):
        os.makedirs("./data/" + cell_type)

    # Loop through experiments
    for exp in metadata["@graph"]:
        # Get metadata and download files for this experiment
        expID = exp["accession"]
        fetch_data(expID)





