#!/usr/bin/env python3
"""
Run dimensionality reduction algorithms on a PLINK SNP data matrix with
0, 1, 2 genotype encodings (i.e. the number of minor alleles). Samples as rows
and SNPs as columns.

Arguments:
  path (str): Path to working directory
  Xtrain (str): Name of training data file in working directory
  Xval (str): Name of validation data file in working directory
  label (list): Name of label column to drop
  save (str): Name of file to save output matrix as
  alg (str): Algorithm to run (pca/svd)

Returns:
  A data matrix with reduced dimension for each method
  (PCA and SVD).
"""

import sys,os
import argparse
import pickle
import datatable as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import itertools as IT
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.decomposition import NMF
from pathlib import Path


def uniquify(path, sep = '_'):
  """
  Function to generate unique file names.
  Source: https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
  """
  def name_sequence():
    count = IT.count()
    yield ''
    while True:
      yield '{s}{n:d}'.format(s = sep, n = next(count))
  orig = tempfile._name_sequence 
  with tempfile._once_lock:
    tempfile._name_sequence = name_sequence()
    path = os.path.normpath(path)
    dirname, basename = os.path.split(path)
    filename, ext = os.path.splitext(basename)
    fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
    tempfile._name_sequence = orig
  return filename


def prep_df(df, labels):
  # Drop extra columns and convert to numpy array
  label = df[labels.split(",")] # save label columns
  df = df.drop(labels.split(","), axis=1)
  ID = df.photoid # save instance IDs
  df.set_index("photoid", inplace=True) # set sample IDs as index
  df_n = df.to_numpy() # convert to numpy array

  # Center genotypes
  scaler = StandardScaler(with_mean=True, with_std=True)
  dfs = scaler.fit_transform(df_n) # centered only
  return dfs, ID, label


def run_IPCA(df, n):
  """
  Incremental PCA
  Input:
    df (numpy array): numpy array (do not center)
    n (int): number of components
  Returns:
    [1] A matrix of n samples and n components.
  """
  print("Computing PCs...")
  ipca = IncrementalPCA(n_components = n)
  ipca.fit(df)
  print("No. components:", ipca.components_.shape)
  print("Explained Var:", ipca.explained_variance_ratio_)
  return ipca


if __name__ == "__main__":
  # Argument parser
  parser = argparse.ArgumentParser(
    description="Run dimension reduction techniques on genotype data")
  req_group = parser.add_argument_group(title="Required Input")
  req_group.add_argument(
    "-path", type=str, help="Path to working directory", required=True)
  req_group.add_argument(
    "-label", type=str, help="Name of label column to drop", required=True)
  req_group.add_argument(
    "-save", type=str,
    help="Name of file to save output matrix as", required=True)
  req_group.add_argument(
    "-split", type=str, help=
    "y/n split file into training and validation sets before applying dimension reduction",
    required=True)
  req_group.add_argument(
    "-n", type=int, help="Number of components for PCA", required=True)
  opt_group = parser.add_argument_group(title="Optional Input")
  opt_group.add_argument(
    "-alg", type=str,
    help="Algorithm to run (pca/ipca/svd/nmf)", default="ipca")
  opt_group.add_argument(
    "-Xtrain", type=str,
    help="Name of training data file in working directory")
  opt_group.add_argument(
    "-Xval", type=str,
    help="Name of validation data file in working directory", default="")
  opt_group.add_argument(
    "-Xinfer", type=str,
    help="Name of inference data file in working directory", default="")
  opt_group.add_argument(
    "-pcaModelDir", type=str,
    help="ddir with PCA models to transform inference data", default="")
  if len(sys.argv)==1:
    parser.print_help()
    sys.exit(0)
  args = parser.parse_args()

  # Set working directory
  #os.chdir(args.path)

  if args.Xval != "":
    # Read in data
    df_train = dt.fread(args.Xtrain)
    df_train = df_train.to_pandas()
    df_val = dt.fread(args.Xval)
    df_val = df_val.to_pandas()
    
    # Drop extra columns and convert to numpy array
    dfs_train, ID_train, label_train = prep_df(df_train, args.label)
    dfs_val, ID_val, label_val = prep_df(df_val, args.label)

    ## Run Incremental PCA
    if args.alg=="ipca":
      ipca = run_IPCA(dfs_train, args.n)
      pickle.dump(ipca, open(f"{args.save}_PCA_model.pkl", "wb"))
      out_train = ipca.transform(dfs_train)
      out_train = pd.DataFrame(out_train)
      out_train.insert(0, "photoid", ID_train) # insert sample IDs
      for col in args.label.split(","):
        out_train.insert(1, col, label_train[col]) # insert label column
      out_train.to_csv(f"{args.save}_train_PCA.csv", index=False, chunksize=1000)
      out_val = ipca.transform(dfs_val)
      out_val = pd.DataFrame(out_val)
      out_val.insert(0, "photoid", ID_val)
      for col in args.label.split(","):
        out_val.insert(0, col, label_val[col]) # insert label column
      out_val.to_csv(f"{args.save}_valid_PCA.csv", index=False, chunksize=1000)
  
  elif args.Xinfer != "": # Transform the inference data using a previously fitted model
    # Read in data
    df_infer = dt.fread(os.path.join(args.path, args.Xinfer))
    df_infer = df_infer.to_pandas()

    #print(df_infer.shape)

    # Drop extra columns and convert to numpy array
    dfs_infer, ID_infer, label_infer = prep_df(df_infer, args.label)
    
    #print(dfs_infer.shape)

    ## Run Incremental PCA
    if args.alg=="ipca":
      pca_files  = Path(args.pcaModelDir).iterdir()
      for pca_file in pca_files:
        if str(pca_file).find('_PCA_model') != -1:
          taxon_level = str(pca_file).split('/')[-1].split('_')[0]

          ipca = pickle.load(open(pca_file, 'rb')) # load fitted model
          out_infer = ipca.transform(dfs_infer)
          out_infer = pd.DataFrame(out_infer)
          out_infer.insert(0, "photoid", ID_infer)
          #for col in args.label.split(","):
          #  out_infer.insert(1, col, label_infer[col]) # insert label column
          out_infer.to_csv(os.path.join(args.path,f"{taxon_level}_{args.save}_PCA.csv"), 
                           index=False, chunksize=1000)

  else:
    # Read in data
    df = dt.fread(args.Xtrain)
    df = df.to_pandas()
    dfs, ID, label = prep_df(df, args.label)

    ## Run Incremental PCA
    if args.alg=="ipca":
      ipca = run_IPCA(dfs, args.n)
      pickle.dump(ipca, open(f"{args.save}_PCA_model.pkl", "wb"))
      out = ipca.transform(dfs)
      out = pd.DataFrame(out)
      out.insert(0, "photoid", ID)
      for col in args.label.split(","):
        out.insert(1, col, label[col]) # insert label column
      out.to_csv(os.path.join(args.path,f"{args.save}_PCA.csv"), index=False, chunksize=1000)