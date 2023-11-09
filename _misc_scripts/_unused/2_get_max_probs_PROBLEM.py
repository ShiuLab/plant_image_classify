#!/usr/bin/python3
"""
For each photo instance, get the max prediction probability for each label in
and filter based on a set threshold.

Input:
  A comma-separated list of thresholds in the order class, order, family,
  genus, and species. 
  E.g., python 2_get_max_probs.py .5,.7,.5,.6,.8

Returns:
  [1] A CSV file containing the max probabilities for each label and each photoid instance.
  [2] A CSV file containing the backfilled labels in file [1], joint probabilities were not adjusted.

6/2/23: Modified by Shiu
"""
__author__ = "Kenia Segura Abá"

import sys, argparse
import datatable as dt
import pandas as pd
import numpy as np
from pathlib import Path
pd.set_option('display.max_columns', None)

def parse_arguments():
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "-i", "--input_dir",
    type=str,
    help="input dir with logistic regression probability files",
    required=True,
  )
  parser.add_argument(
    "-o", "--output_dir",
    type=str,
    help="directory to generate outputs",
    required=True,
  )
  parser.add_argument(
    "-e", "--meta_data_dir",
    type=str,
    help="directory with encodings",
    required=True,
  )
  parser.add_argument(
    "-t", "--thresholds",
    type=str,
    default=".5,.5,.5,.5,.5",
    help="threshold probabilities for taxon levels (default: .5,.5,.5,.5,.5)",
    required=False,
  )
  return parser.parse_args()

if __name__ == "__main__":
  # Arguments help
  #if len(sys.argv) == 1:
  #  print("Threshold (float) required")
  #  sys.exit()
  args         = parse_arguments()
  input_dir    = Path(args.input_dir)
  output_dir   = Path(args.output_dir)
  meta_data_dir = Path(args.meta_data_dir)
  thresh       = args.thresholds

  print("Read prediction probabilities")
  class_preds = order_preds = family_preds = genus_preds = species_preds = None
  for file in input_dir.iterdir():
    fstr = str(file)
    if fstr.find("logisticreg.csv") != -1:
      print('reading file:', fstr)
      taxon_level = fstr.split("/")[-1].split("_")[2]
      print(" ", taxon_level, end=":")
      if taxon_level == "class":
         class_preds = pd.read_csv(file, index_col=0)
         print(class_preds.shape) 
      elif taxon_level == "order":
         order_preds = pd.read_csv(file, index_col=0)
         print(order_preds.shape) 
      elif taxon_level == "family":
         family_preds = pd.read_csv(file, index_col=0)
         print(family_preds.shape) 
      elif taxon_level == "genus":
         genus_preds = pd.read_csv(file, index_col=0)
         print(genus_preds.shape) 
      elif taxon_level == "species":
         species_preds = pd.read_csv(file, index_col=0)
         print(species_preds.shape) # (4867, 1451)

  # Set a threshold for filtering max probabilities for each label
  print("Set thresholds")
  thr_class, thr_order, thr_family, thr_genus, thr_species = \
                                          list(map(float, thresh.split(",")))
  print(" ", thr_class, thr_order, thr_family, thr_genus, thr_species)

  class_encod = order_encod = order_encod = family_encod = genus_encod = \
    species_encod = None
  print("Read label encoding files")
  for file in meta_data_dir.iterdir():
    fstr = str(file)
    if fstr.find("encoding.txt") != -1:
      taxon_level = fstr.split("/")[-1].split("_")[1].split("-")[0]
      if taxon_level == "class":
        class_encod = pd.read_csv(file, sep=" ", header=None)
        print(" ", taxon_level, class_encod.shape)
      elif taxon_level == "order":
        order_encod = pd.read_csv(file, sep=" ", header=None)
        print(" ", taxon_level, order_encod.shape)
      elif taxon_level == "family":
        family_encod = pd.read_csv(file, sep=" ", header=None)
        print(" ", taxon_level, family_encod.shape)
      elif taxon_level == "genus":
        genus_encod = pd.read_csv(file, sep=" ", header=None)
        print(" ", taxon_level, genus_encod.shape)
      elif taxon_level == "species":
        species_encod = pd.read_csv(file, sep=" ", header=None)
        print(" ", taxon_level, species_encod.shape)

  # SHS: assetion test removed, encoding size and prediction size may not match
  #  because some labels are not used for training because there are too few
  #  images

  print("Read taxa parent-child relationships")
  meta = pd.read_csv(meta_data_dir / "taxa_parent_child.txt", sep="\t")
  print(" ", meta.shape)

  #### Starting with the top-level, class, get the max prediction probability 
  #    and class level for each photoid
  print("Start with class level")

  max_class_preds = class_preds.idxmax(axis=1) # class level of max probability
  max_class_preds = pd.concat([class_preds.apply(max, axis=1), 
                               max_class_preds], axis=1) # max probability
  max_class_preds.columns = ["max_prob_class", "class_lvl"]
 
  # Check 2: Ensure that the photoid instances are all there
  try:
    assert max_class_preds.shape[0]==class_preds.shape[0], \
      "Not all the photoid instances are in class_preds"
  except AssertionError:
    pass
  #### Get the orders for each photoid and corresponsing max probability class level
  max_class_preds["max_prob_order"] = None
  max_class_preds["order_lvl"] = None
  print(" ", max_class_preds.shape)
  # get the child orders for each class level
  class_lvl_orders = {key: None for key in max_class_preds["class_lvl"].unique()} 
  meta_sub = meta[meta.taxon_level=="class"] # subset the metadata
  # Check 3: Ensure that all the class levels are in the metadata
  try:
    assert all(key in meta_sub.label.values \
      for key in list(map(int, list(class_lvl_orders.keys())))), \
      f"Not all class levels are in the metadata.  Metadata is missing " +\
      f"{len(meta_sub.label)-len(list(class_lvl_orders.keys()))} levels"
  except AssertionError:
    pass
  # Get the orders for each class
  for class_lvl in class_lvl_orders:
    if int(class_lvl) in meta_sub.label.values:
      class_lvl_orders[class_lvl] = str(list(meta_sub.loc[meta_sub.label==int(class_lvl), "child_labels"].values)[0]).split(",")
  # Iterate through all the photoids in max_class_preds
  for row in range(max_class_preds.shape[0]):
    photoid = max_class_preds.index[row] # photo id
    # Check 4: Ensure max probability of photoid's class_lvl meets the threshold
    if max_class_preds.iloc[row, 0] >= thr_class:
      # Check 5: Ensure photoid exists in order_preds
      if photoid in order_preds.index:
        # Get the corresponding order levels for the class_lvl of this photoid
        class_lvl = max_class_preds.iloc[row,1]
        # Check 6: Skip over class_lvl with no children in metadata
        if class_lvl_orders[class_lvl] is not None:
          cols = list(map(str, class_lvl_orders[class_lvl]))
          # Check 7: Ensure only the columns in cols are in order_preds
          cols = [col for col in cols if col in order_preds.columns]
          # Check 8: Ensure cols is not empty
          if len(cols) != 0:
            # Get the order level for the max probability
            #print(order_preds.loc[photoid, cols].idxmax(axis=1))
            max_order_idx = order_preds.loc[photoid, cols].idxmax(axis=0)
            max_order_prob = order_preds.loc[photoid, max_order_idx]
            # Check 9: Ensure the order level matches the max order probability
            # print(order_preds.loc[photoid, max_order_idx]==max_order_prob)
            try:
              assert order_preds.loc[photoid, max_order_idx]==max_order_prob, "The max order level does not match the max prediction probability"
            except AssertionError as e:
              print(e)
            # Check 10: Ensure the order max probability meets the threshold
            if max_order_prob >= thr_order:
              # Update max_class_preds
              max_class_preds.loc[photoid, "max_prob_order"] = max_order_prob
              max_class_preds.loc[photoid, "order_lvl"] = max_order_idx
    else:
      # If the class probability doesn't meet the threshold, start with order
      photoid = max_class_preds.index[row]
      # Get the corresponding order levels for the class_lvl of this photoid
      class_lvl = max_class_preds.iloc[row,1]
      child_orders = class_lvl_orders[class_lvl]
      if child_orders is not None:
        cols = [col for col in child_orders if col in order_preds.columns]
        # Calculate the max probability of order within the children of this class_lvl
        if photoid in order_preds.index:
          max_order_preds = order_preds[cols].idxmax(axis=1) # order level max probability
          max_order_preds = pd.concat([order_preds[cols].apply(max, axis=1), max_order_preds], axis=1) # max probability
          max_order_preds.columns = ["max_prob_order", "order_lvl"]
          max_order_idx = max_order_preds.loc[photoid, "order_lvl"]
          # print(f'CHECK {photoid}, {max_order_idx}')
          # If the probability meets the threshold, add to max_class_preds
          if max_order_preds.loc[photoid, "max_prob_order"] >= thr_order:
            max_class_preds.loc[photoid, "max_prob_order"] = max_order_preds.loc[photoid, "max_prob_order"]
            max_class_preds.loc[photoid, "order_lvl"] = max_order_idx
      
  print("Start with order level")
  #### Get the families for each photoid and corresponsing max probability order level
  max_class_preds["max_prob_family"] = None
  max_class_preds["family_lvl"] = None
  order_lvl_families = {key: None for key in max_class_preds["order_lvl"].unique()} # get the child families for each order level
  meta_sub = meta[meta.taxon_level=="order"] # subset the metadata
  print(" ", max_class_preds.shape)
  
  # Remove NoneType key
  if None in order_lvl_families:
    del order_lvl_families[None]
  # Check 11: Ensure that all the order levels are in the metadata
  try:
    assert all(key in meta_sub.label.values for key in list(map(int, list(order_lvl_families.keys())))), f"Not all order levels are in the metadata. Metadata is missing {len(meta_sub.label)-len(list(order_lvl_families.keys()))} levels"
  except AssertionError:
    pass
  # Get the families for each order
  for order_lvl in order_lvl_families:
    if int(order_lvl) in meta_sub.label.values:
      order_lvl_families[order_lvl] = str(list(meta_sub.loc[meta_sub.label==int(order_lvl), "child_labels"].values)[0]).split(",")
  # Iterate through all the photoids in max_class_preds
  for row in range(max_class_preds.shape[0]):
    photoid = max_class_preds.index[row] # photo id
    # Check 12: Ensure max probability of photoid's order_lvl meets the threshold
    # print(max_class_preds.iloc[row, 2])
    if max_class_preds.iloc[row, 2] != None and max_class_preds.iloc[row, 2] >= thr_order:
      # Check 13: Ensure photoid exists in family_preds
      if photoid in family_preds.index:
        # Get the corresponding family levels for the order_lvl of this photoid
        order_lvl = max_class_preds.iloc[row,3]
        # Check 14: Skip over order_lvl with no children in metadata
        if order_lvl_families[order_lvl] is not None:
          cols = list(map(str, order_lvl_families[order_lvl]))
          # Check 15: Ensure only the columns in cols are in family_preds
          cols = [col for col in cols if col in family_preds.columns]
          # Check 16: Ensure cols is not empty
          if len(cols) != 0:
            # Get the family level for the max probability
            max_family_idx = family_preds.loc[photoid, cols].idxmax(axis=0)
            max_family_prob = family_preds.loc[photoid, max_family_idx]
            # Check 17: Ensure the family level matches the max family probability
            #print(family_preds.loc[photoid, max_family_idx]==max_family_prob)
            try:
              assert family_preds.loc[photoid, max_family_idx]==max_family_prob, "The max family level does not match the max prediction probability"
            except AssertionError as e:
              print(e)
            # Check 18: Ensure the family max probability meets the threshold
            if max_family_prob >= thr_family:
              # Update max_class_preds
              max_class_preds.loc[photoid, "max_prob_family"] = max_family_prob
              max_class_preds.loc[photoid, "family_lvl"] = max_family_idx
    else:
      # If the order probability doesn't meet the threshold, start with family
      photoid = max_class_preds.index[row]
      # Get the corresponding family levels for the order_lvl of this photoid
      order_lvl = max_class_preds.iloc[row,3]
      if order_lvl is not None:
        print('CHECK', photoid)
        print('CHECK', max_class_preds.head())
        child_families = order_lvl_families[order_lvl]
        if child_families is not None:
          cols = [col for col in child_families if col in family_preds.columns]
          # Calculate the max probability of family within the children of this order_lvl
          if photoid in family_preds.index:
            max_family_preds = family_preds[cols].idxmax(axis=1) # family level max probability
            max_family_preds = pd.concat([family_preds[cols].apply(max, axis=1), max_family_preds], axis=1) # max probability
            max_family_preds.columns = ["max_prob_family", "family_lvl"]
            max_family_idx = max_family_preds.loc[photoid, "family_lvl"]
            # print(f'CHECK {photoid}, {max_family_idx}')
            # If the probability meets the threshold, add to max_class_preds
            if max_family_preds.loc[photoid, "max_prob_family"] >= thr_family:
              max_class_preds.loc[photoid, "max_prob_family"] = max_family_preds.loc[photoid, "max_prob_family"]
              max_class_preds.loc[photoid, "family_lvl"] = max_family_idx

  print("Start with family level")
  #### Get the genera for each photoid and corresponsing max probability family level
  max_class_preds["max_prob_genus"] = None
  max_class_preds["genus_lvl"] = None
  family_lvl_genera = {key: None for key in max_class_preds["family_lvl"].unique()} # get the child genera for each family level
  meta_sub = meta[meta.taxon_level=="family"] # subset the metadata
  print(" ", max_class_preds.shape)

  # Remove NoneType key
  if None in family_lvl_genera:
    del family_lvl_genera[None]
  # Check 19: Ensure that all the family levels are in the metadata
  try:
    assert all(key in meta_sub.label.values for key in list(map(int, list(family_lvl_genera.keys())))), f"Not all family levels are in the metadata. Metadata is missing {len(meta_sub.label)-len(list(family_lvl_genera.keys()))} levels"
  except AssertionError:
    pass
  # Get the genera for each family
  for family_lvl in family_lvl_genera:
    if int(family_lvl) in meta_sub.label.values:
      family_lvl_genera[family_lvl] = list(meta_sub.loc[meta_sub.label==int(family_lvl), "child_labels"].values)[0].split(",")
  # Iterate through all the photoids in max_class_preds
  for row in range(max_class_preds.shape[0]):
    photoid = max_class_preds.index[row] # photo id
    # Check 20: Ensure max probability of photoid's family_lvl meets the threshold
    # print(max_class_preds.iloc[row, 4])
    if max_class_preds.iloc[row, 4] != None and max_class_preds.iloc[row, 4] >= thr_family:
      # Check 21: Ensure photoid exists in genus_preds
      if photoid in genus_preds.index:
        # Get the corresponding genus levels for the family_lvl of this photoid
        family_lvl = max_class_preds.iloc[row,5]
        # Check 22: Skip over family_lvl with no children in metadata
        if family_lvl_genera[family_lvl] is not None:
          cols = list(map(str, family_lvl_genera[family_lvl]))
          # Check 23: Ensure only the columns in cols are in genus_preds
          cols = [col for col in cols if col in genus_preds.columns]
          # Check 24: Ensure cols is not empty
          if len(cols) != 0:
            # Get the genus level for the max probability
            max_genus_idx = genus_preds.loc[photoid, cols].idxmax(axis=0)
            max_genus_prob = genus_preds.loc[photoid, max_genus_idx]
            # Check 25: Ensure the genus level matches the max family probability
            #print(genus_preds.loc[photoid, max_genus_idx]==max_genus_prob)
            try:
              assert genus_preds.loc[photoid, max_genus_idx]==max_genus_prob, "The max genus level does not match the max prediction probability"
            except AssertionError as e:
              print(e)
            # Check 26: Ensure the genus max probability meets the threshold
            if max_genus_prob >= thr_genus:
              # Update max_class_preds
              max_class_preds.loc[photoid, "max_prob_genus"] = max_genus_prob
              max_class_preds.loc[photoid, "genus_lvl"] = max_genus_idx
    else:
      # If the family probability doesn't meet the threshold, start with genus
      photoid = max_class_preds.index[row]
      # Get the corresponding genus levels for the family_lvl of this photoid
      family_lvl = max_class_preds.iloc[row,5]
      if family_lvl is not None:
        child_genera = family_lvl_genera[family_lvl]
        if child_genera is not None:
          cols = [col for col in child_genera if col in genus_preds.columns]
          # Calculate the max probability of genus within the children of this family_lvl
          if photoid in genus_preds.index:
            max_genus_preds = genus_preds[cols].idxmax(axis=1) # genus level max probability
            max_genus_preds = pd.concat([genus_preds[cols].apply(max, axis=1), max_genus_preds], axis=1) # max probability
            max_genus_preds.columns = ["max_prob_genus", "genus_lvl"]
            max_genus_idx = max_genus_preds.loc[photoid, "genus_lvl"]
            # print(f'CHECK {photoid}, {max_genus_idx}')
            # If the probability meets the threshold, add to max_class_preds
            if max_genus_preds.loc[photoid, "max_prob_genus"] >= thr_genus:
              max_class_preds.loc[photoid, "max_prob_genus"] = max_genus_preds.loc[photoid, "max_prob_genus"]
              max_class_preds.loc[photoid, "genus_lvl"] = max_genus_idx

  print("Start with genus level")
  #### Get the species for each photoid and corresponsing max probability genus level
  max_class_preds["max_prob_species"] = None
  max_class_preds["species_lvl"] = None
  genus_lvl_species = {key: None for key in max_class_preds["genus_lvl"].unique()} # get the child species for each genus level
  meta_sub = meta[meta.taxon_level=="genus"] # subset the metadata
  print(" ", max_class_preds.shape)

  # Remove NoneType key
  if None in genus_lvl_species:
    del genus_lvl_species[None]
  # Check 27: Ensure that all the genus levels are in the metadata
  try:
    assert all(key in meta_sub.label.values for key in list(map(int, list(genus_lvl_species.keys())))), f"Not all genus levels are in the metadata. Metadata is missing {len(meta_sub.label)-len(list(genus_lvl_species.keys()))} levels"
  except AssertionError:
    pass
  # Get the species for each genus
  for genus_lvl in genus_lvl_species:
    if int(genus_lvl) in meta_sub.label.values:
      genus_lvl_species[genus_lvl] = list(meta_sub.loc[meta_sub.label==int(genus_lvl), "child_labels"].values)[0].split(",")
  # Iterate through all the photoids in max_class_preds
  for row in range(max_class_preds.shape[0]):
    photoid = max_class_preds.index[row] # photo id
    # Check 28: Ensure max probability of photoid's genus_lvl meets the threshold
    # print(max_class_preds.iloc[row, 6])
    if max_class_preds.iloc[row, 6] != None and max_class_preds.iloc[row, 6] >= thr_genus:
      # Check 29: Ensure photoid exists in species_preds
      if photoid in species_preds.index:
        # Get the corresponding species levels for the genus_lvl of this photoid
        genus_lvl = max_class_preds.iloc[row,7]
        # Check 30: Skip over genus_lvl with no children in metadata
        if genus_lvl_species[genus_lvl] is not None:
          cols = list(map(str, genus_lvl_species[genus_lvl]))
          # Check 31: Ensure only the columns in cols are in species_preds
          cols = [col for col in cols if col in species_preds.columns]
          # Check 32: Ensure cols is not empty
          if len(cols) != 0:
            # Get the species level for the max probability
            max_species_idx = species_preds.loc[photoid, cols].idxmax(axis=0)
            max_species_prob = species_preds.loc[photoid, max_species_idx]
            # Check 33: Ensure the species level matches the max species probability
            #print(species_preds.loc[photoid, max_species_idx]==max_species_prob)
            try:
              assert species_preds.loc[photoid, max_species_idx]==max_species_prob, "The max species level does not match the max prediction probability"
            except AssertionError as e:
              print(e)
            # Check 34: Ensure the species max probability meets the threshold
            if max_species_prob >= thr_species:
              # Update max_class_preds
              max_class_preds.loc[photoid, "max_prob_species"] = max_species_prob
              max_class_preds.loc[photoid, "species_lvl"] = max_species_idx
    else:
      # If the genus probability doesn't meet the threshold, start with species
      photoid = max_class_preds.index[row]
      # Get the corresponding species levels for the genus_lvl of this photoid
      genus_lvl = max_class_preds.iloc[row,7]
      if genus_lvl is not None:
        print(genus_lvl_species)
        child_species = genus_lvl_species[genus_lvl]
        if child_species is not None:
          cols = [col for col in child_species if col in genus_preds.columns]
          # Calculate the max probability of species within the children of this genus_lvl
          if photoid in species_preds.index:
            max_species_preds = species_preds[cols].idxmax(axis=1) # species level max probability
            max_species_preds = pd.concat([species_preds[cols].apply(max, axis=1), max_species_preds], axis=1) # max probability
            max_species_preds.columns = ["max_prob_species", "species_lvl"]
            max_species_idx = max_species_preds.loc[photoid, "species_lvl"]
            # print(f'CHECK {photoid}, {max_species_idx}')
            # If the probability meets the threshold, add to max_class_preds
            if max_species_preds.loc[photoid, "max_prob_species"] >= thr_species:
              max_class_preds.loc[photoid, "max_prob_species"] = max_species_preds.loc[photoid, "max_prob_species"]
              max_class_preds.loc[photoid, "species_lvl"] = max_species_idx

  print("Set class prob < threshold to None")
  # Set class probabilities less than threshold to None
  max_class_preds.loc[max_class_preds["max_prob_class"] < thr_class, "class_lvl"] = None
  max_class_preds.loc[max_class_preds["max_prob_class"] < thr_class, "max_prob_class"] = None

  # Make probability columns numerical
  print("Prob to float")
  max_class_preds["max_prob_order"] = max_class_preds.max_prob_order.astype("float64")
  max_class_preds["max_prob_family"] = max_class_preds.max_prob_family.astype("float64")
  max_class_preds["max_prob_genus"] = max_class_preds.max_prob_genus.astype("float64")
  max_class_preds["max_prob_species"] = max_class_preds.max_prob_species.astype("float64")
  
  print("Calculate joint probabilities")
  max_class_preds["joint_prob"] = 0
  # max_class_preds["mean_prob"] = 0 #################################################################### average
  for row in range(max_class_preds.shape[0]):
    photoid = max_class_preds.index[row]
    values = max_class_preds.iloc[row,:].dropna()
    values = values.loc[~values.index.str.contains("_lvl")] # drop *_lvl columns
    values = values.loc[~values.index.str.contains("joint_")] # drop joint_prob column
    max_class_preds.loc[photoid, "joint_prob"] = np.prod(values.to_list())
    # max_class_preds.loc[photoid, "mean_prob"] = np.mean(values.to_list()) ############################# average
    del values

  # Change column order
  max_class_preds.columns = ['max_prob_class', 'class', 'max_prob_order',
    'order', 'max_prob_family', 'family', 'max_prob_genus',
    'genus', 'max_prob_species', 'species', 'joint_prob', 'mean_prob'] ############################# average
  max_class_preds = max_class_preds[['class', 'order', 'family', 'genus', 'species', 
    'max_prob_class', 'max_prob_order', 'max_prob_family', 'max_prob_genus',
    'max_prob_species', 'joint_prob']]#, 'mean_prob']] ############################# average

  # Save files
  print("Saving file of max prediction probabilities...")
  start_folder_name = str(input_dir).split("/")[-1].split("_")[1]
  debug_file = input_dir / f"{start_folder_name}_probs.{'_'.join(thresh.split(','))}.csv"
  # debug_file = output_dir / f"{start_folder_name}_probs.{'_'.join(thresh.split(','))}.csv"
  
  print("  master folder name:", start_folder_name)
  print("  debug file:", debug_file)

  print("Back filling missing taxa")
  # Back fill from the deepest taxa level with above threshold prediction,
  # Output:
  #   photoid,class,order,family,genus,species,JOINT_PROB_FOR_ABOVE_THRE_TAXA
  # print(max_class_preds.dtypes)
  backfilled = max_class_preds[["class", "order", "family", "genus", "species"]]
  # print(backfilled.dtypes)
  for row in range(backfilled.shape[0]):
    photoid = backfilled.index[row]
    for col in range(backfilled.shape[1]-1, 0, -1): # iterate starting from deepest taxa level
      # Cases where the first non NaN column (deepest taxa level) and the parent taxa is NaN
      if (backfilled.iloc[row, col]!=None) & (backfilled.iloc[row, col-1] is None):
          if backfilled.columns[col]=="order":
            order_lvl = int(backfilled.iloc[row, col])
            parent = meta.loc[meta["taxon_level"]=="class"].copy(deep=True)
            parent["child_labels"] = parent.loc[:,"child_labels"].str.split(",").apply(lambda x: [int(lvl) for lvl in x])
            parent = parent.loc[parent.child_labels.apply(lambda x: order_lvl in x), "label"].values[0]
            # print('COL', backfilled.columns[col], backfilled.iloc[row, col])
            # print('COL-1', backfilled.columns[col-1], backfilled.iloc[row, col-1])
            # print('CHECK', parent)
            # print(backfilled.iloc[row, col-1])
            backfilled.iloc[row, col-1] = parent
            # print(backfilled.iloc[row, col-1])
          elif backfilled.columns[col]=="family":
            family_lvl = int(backfilled.iloc[row, col])
            parent = meta.loc[meta["taxon_level"]=="order"].copy(deep=True)
            parent["child_labels"] = parent.loc[:,"child_labels"].str.split(",").apply(lambda x: [int(lvl) for lvl in x])
            parent = parent.loc[parent.child_labels.apply(lambda x: family_lvl in x), "label"].values[0]
            # print('COL', backfilled.columns[col], backfilled.iloc[row, col])
            # print('COL-1', backfilled.columns[col-1], backfilled.iloc[row, col-1])
            # print('CHECK', parent)
            # print(backfilled.iloc[row, col-1])
            backfilled.iloc[row, col-1] = parent
            # print(backfilled.iloc[row, col-1])
          elif backfilled.columns[col]=="genus":
            genus_lvl = int(backfilled.iloc[row, col])
            parent = meta.loc[meta["taxon_level"]=="family"].copy(deep=True)
            parent["child_labels"] = parent.loc[:,"child_labels"].str.split(",").apply(lambda x: [int(lvl) for lvl in x])
            parent = parent.loc[parent.child_labels.apply(lambda x: genus_lvl in x), "label"].values[0]
            # print('COL', backfilled.columns[col], backfilled.iloc[row, col])
            # print('COL-1', backfilled.columns[col-1], backfilled.iloc[row, col-1])
            # print('CHECK', parent)
            # print(backfilled.iloc[row, col-1])
            backfilled.iloc[row, col-1] = parent
            # print(backfilled.iloc[row, col-1])
          elif backfilled.columns[col]=="species": 
            species_lvl = int(backfilled.iloc[row, col])
            parent = meta.loc[meta["taxon_level"]=="genus"].copy(deep=True)
            parent["child_labels"] = parent.loc[:,"child_labels"].str.split(",").apply(lambda x: [int(lvl) for lvl in x])
            parent = parent.loc[parent.child_labels.apply(lambda x: species_lvl in x), "label"].values[0]
            # print('COL', backfilled.columns[col], backfilled.iloc[row, col])
            # print('COL-1', backfilled.columns[col-1], backfilled.iloc[row, col-1])
            # print('CHECK', parent)
            # print(backfilled.iloc[row, col-1])
            backfilled.iloc[row, col-1] = parent
            # print(backfilled.iloc[row, col-1])
  # Add the joint probabilities column
  backfilled.insert(backfilled.shape[1], "joint_prob", max_class_preds[["joint_prob"]])
  # backfilled.insert(backfilled.shape[1], "mean_prob", max_class_preds[["mean_prob"]]) ################################ average
  
  print("Generate outputs")
  #backfilled_file = output_dir / f"{start_folder_name}_backfilled_probs.{'_'.join(thresh.split(','))}.csv"
  backfilled_file = input_dir / f"{start_folder_name}_backfilled.csv"
  print("  backfilled file:", backfilled_file)

  #print("Saving final file without backfilling missing taxa")
  max_class_preds.to_csv(debug_file)
  #max_class_preds.to_csv(final_file)
  backfilled.to_csv(backfilled_file)

  # Genreate final output with translated taxa names
  final_file = output_dir / f"{start_folder_name}.csv"
  print("  final file:", final_file)
  # Read in the taxa names
  taxa_name_file = meta_data_dir / "taxa_name_to_label.txt"
  tdict = {} # {taxon_level:{taxon_label:taxon_name}}
  with open(taxa_name_file, "r") as f:
    for line in f:
      if line != "\n":
        [taxon_level, taxon_name, taxon_label] = line.strip().split("\t")
        if taxon_level not in tdict:
          tdict[taxon_level] = {taxon_label:taxon_name}
        elif taxon_label not in tdict[taxon_level]:
          tdict[taxon_level][taxon_label] = taxon_name
        else:
          print("ERR: duplicate taxon label", taxon_level, taxon_label)

  t1 = "Plantae"            # kingdom
  t9 = "ML_plant_ensemble"  # identification_method
  print('CHECK', backfilled.head())
  with open(final_file, "w") as f:
    f.write("photoid,kingdom,phyla,class,order,family,genus,species_epithet,"+\
            "taxon_rank,identification_method,confidence_percent\n")
    for row in range(backfilled.shape[0]):
      row_ser = backfilled.iloc[row,:]
      row_val = row_ser.values.tolist()
      print('CHECK', row_val)
      t0      = row_ser.index                          # photoid
      t2      = tdict["phylum"][row_val[0]]            # phyla
      t3      = row_ser["class"]                       # class
      t4      = row_ser["order"]                       # order
      t5      = row_ser["family"]                      # family
      t6      = row_ser["genus"]                       # genus
      if row_val[4] != None:
        t7    = row_val[4].split(" ")[0]               # species epithet
      else:
        t7    = "None"
      t10     = '{0:.2f}'.format(float(row_val[5])*100) # confidence percent
      #t11     =                                        # mean probability
      print('CHECK2', f"{t0},{t1},{t2},{t3_6},{t7},{t9},{t10}")#, {t11}")
      f.write(f"{t1},{t2},{t3_6},{t7},{t9},{t10}\n")