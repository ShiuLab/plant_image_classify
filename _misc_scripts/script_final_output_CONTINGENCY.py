#!/usr/bin/python3
"""
For each photo instance, get the max prediction probability for each label in
and filter based on a set threshold.

6/2/23: Started by Shiu  
6/4/23: Modified by Kenia Segura Abá

Ref:
- https://stackoverflow.com/questions/68622484/find-the-max-value-at-each-row-pandas-data-frame
- https://stackoverflow.com/questions/48896900/output-a-single-row-in-pandas-to-an-array
- https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
- https://stackoverflow.com/questions/43643506/select-columns-based-on-columns-names-containing-a-specific-string-in-pandas

"""
import sys, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

def parse_arguments():
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "-i", "--input_dir",
    type=str,
    default='/mnt/research/xprize23/plants_essential/example_data/tmp_C201',
    help="input dir with logistic regression probability files",
    required=False,
  )
  parser.add_argument(
    "-o", "--output_dir",
    type=str,
    default='/mnt/research/xprize23/plants_essential/example_data/',
    help="directory to generate outputs",
    required=False,
  )
  parser.add_argument(
    "-m", "--metadata_dir",
    type=str,
    default='/mnt/research/xprize23/plants_essential/meta_data',
    help="directory with encodings",
    required=False,
  )
  parser.add_argument(
    "-t", "--thresholds",
    type=str,
    default=".38,4.5e-2,8e-3,7e-3,2e-3",
    help="threshold probabilities for taxon levels (default: .38,4.5e-2,8e-3,7e-3,2e-3)",
    required=False,
  )
  parser.add_argument(
    "-d","--debug",
    action="store_true",
    help="Output debug file."
  )
  return parser.parse_args()

def read_pred_prob(input_dir, taxon_levels):
  '''
  Read the prediction probability files for each taxon level
  Args:
    input_dir: directory with logistic regression probability files
    taxon_levels: list of taxon levels in order
  Return:
    prob_df_list: list of dataframes for each taxon level
    label_list: list of labels for each taxon level
  '''

  prob_df_list  = []
  label_list    = []

  file_list = ["log_inference_class_effnet.pth",
               "log_inference_family_effnet.pth",
               "log_inference_genus_effnet.pth",
               "log_inference_order_effnetv2-b0_lr1e-3.pth",
               "log_inference_species_effnetv2-b0.pth"]

  for taxon_level in taxon_levels:
    
    for file in file_list:
      if file.find(f"log_inference_{taxon_level}") != -1:
        log_file_name = file
    log_file = input_dir / log_file_name
    df = pd.read_csv(log_file, index_col=0)
    print(f"  {taxon_level}: {df.shape}")
    print(df.columns)
    df.drop(["pred","conf"], axis=1, inplace=True)
    prob_df_list.append(df)
    label_list.append(list(df.columns))

  row_num_list = [df.shape[0] for df in prob_df_list]
  for row_num1, row_num2 in zip(row_num_list[:-1], row_num_list[1:]):
    if row_num1 != row_num2:
      print("ERR: num instances not equal across taxon levels, STOP!")
      sys.exit(0)

  print("  num instances across:", row_num_list[0])

  return prob_df_list, label_list


def get_parent_child(p_c_file):
  '''
  Get the parent-child relationships for each taxon level. Note for c_p_dict,
  the key is parent_taon_level which is a bit counterintuitive.
  Args:
    p_c_file: {parent_taxon_level: {parent: [child1, child2, ...]}
    c_p_file: {parent_taxon_level: {child: parent}
  '''
  p_c_dict = {} 
  c_p_dict = {}
  with open(p_c_file, "r") as f:
    # rid of header
    f.readline()
    for line in f:
      [taxon_level, parent, children] = line.strip().split("\t")
      children = children.split(",")

      # fill P_C dict
      if taxon_level not in p_c_dict:
        p_c_dict[taxon_level] = {parent:children}
      elif parent not in p_c_dict[taxon_level]:
        p_c_dict[taxon_level][parent] = children
      else:
        print("ERR: filling p_c_dict, should not happen")
        sys.exit(0)

      # fill C_P dict
      # [NOTE]: parent_taxon_level is the key
      for child in children:
        if taxon_level not in c_p_dict:
          c_p_dict[taxon_level] = {child:parent}
        elif child not in c_p_dict[taxon_level]:
          c_p_dict[taxon_level][child] = parent
        else:
          print("ERR: filling c_p_dict, should not happen")
          print(f"  taxon_level={taxon_level}, child={child}, parent={parent}")
          sys.exit(0)

  return p_c_dict, c_p_dict


def get_max(df, photoid):
  try:
    max_idx   = df.loc[photoid].values.argmax()
  except ValueError as e:
    print("ERR:", e)
    print("  photoid:", photoid)
    print("  df.loc[photoid]:", df.loc[photoid])
    return "", 0

  max_label = df.columns[max_idx]
  max_p     = df.loc[photoid].values[max_idx]

  return max_label, max_p

def fill_pdict(photoid, pdict, level, labels):
  '''
  Recusrive function for populating pdict
  Args:
    photoid: to get probability values
    pdict: {taxon_level: [max_taxon_label, max_prob]}
    level: 0, 1, 2, 3, 4 correspond to class, order, family, genus, species
    labels: list of target labels for this level
  Return:
    pdict: {taxon_level: {taxon_label: prob}}
  '''
  taxon_level = taxon_dict[level]

  # get max prob and label for this level
  #print(level, taxon_level, prob_df_list[level].shape)
  #print(len(prob_df_list[level].columns))
  
  # here some labels some passed labels are not found because they are not
  # present in the training set.
  local_df = prob_df_list[level]
  # check labels to make sure they are in the df
  checked_labels = []
  for label in labels:
    if label in local_df.columns:
      checked_labels.append(label)

  target_df = local_df[checked_labels]
  #print(target_df)
  max_label, max_p = get_max(target_df, photoid)
  if max_label == "":
    print("ERR: max_label is empty")
    print("1", local_df)
    print("2", taxon_level)
    print("3", labels)
    print("4", checked_labels)
    print("5", target_df)
    sys.exit(0)
  #print(f"level={level}", taxon_level, max_label, max_p)

  # fill pdict
  if taxon_level not in pdict:
    pdict[taxon_level] = [max_label, max_p]

  # traverse to next level unless it is already at the species level
  if level < 4:
    #print(level)
    #print(p_c_dict[taxon_level])
    children = p_c_dict[taxon_level][max_label]
    #print(f"children: {children}")
    pdict = fill_pdict(photoid, pdict, level+1, children)
        
  return pdict

def back_fill(photoid, pdict, level):
  '''
  Recursive function for back filling pdict
  '''
  taxon_level = taxon_dict[level]
  taxon_label = pdict[taxon_level][0]

  if level > 0:
    parent_level = taxon_dict[level-1]

    # Note that parent_level is used here, c_p_dict is keyed by parent_level
    parent_label = c_p_dict[parent_level][taxon_label]
    #print(f"DEBUG: level={level-1}, parent_label={parent_label}")
    #print(prob_df_list[level-1])
    
    parent_prob  = prob_df_list[level-1].loc[photoid][parent_label]
    pdict[parent_level] = [parent_label, parent_prob]

    pdict = back_fill(photoid, pdict, level-1)
  
  return pdict

def get_taxa_name_dict(taxon_name_file):
  '''
  Get the taxon name to label dictionary
  Args:
    taxon_name_file: tab limited with taxon_level, taxon_name, taxon_label
  Return:
    taxon_name_dict: {taxon_level: {taxon_label: taxon_name}}
  '''
  taxon_name_dict = {}
  with open(taxon_name_file, "r") as f:
    # rid of header
    f.readline()
    for line in f:
      #print(line.strip())
      [taxon_level, taxon_name, taxon_label] = line.strip().split("\t")
      if taxon_level not in taxon_name_dict:
        taxon_name_dict[taxon_level] = {taxon_label: taxon_name}
      elif taxon_label not in taxon_name_dict[taxon_level]:
        taxon_name_dict[taxon_level][taxon_label] = taxon_name
      else:
        print("ERR: filling taxon_name_dict, should not happen")
        print("redundant taxon_label:", taxon_label, "taxon_name:", taxon_name)
        sys.exit(0)
  return taxon_name_dict


def get_taxon_prob(pdict):
  # get list of taxon from pdict
  taxon_list = []
  probs_list = []
  for taxon_level in taxon_levels:
    [taxon_label, label_p] = pdict[taxon_level]
    taxon_list.append(taxon_label)
    probs_list.append(label_p)
  joint_prob = np.prod(probs_list)
  root_prob  = np.power(joint_prob, 1/len(probs_list))

  return taxon_list, root_prob, probs_list

if __name__ == "__main__":
  # Arguments help
  #if len(sys.argv) == 1:
  #  print("Threshold (float) required")
  #  sys.exit()
  args         = parse_arguments()
  input_dir    = Path(args.input_dir)
  output_dir   = Path(args.output_dir)
  metadata_dir = Path(args.metadata_dir)
  thresh       = args.thresholds

  output_dir.mkdir(parents=True, exist_ok=True)

  print("Process:", input_dir)

  # hard-coded levels to keep of things
  taxon_levels = ['class', 'order', 'family', 'genus', 'species']
  taxon_dict   = {0:'class', 1:'order', 2:'family', 3:'genus', 4:'species'}

  print("Read prediction probabilities")
  prob_df_list, label_list = read_pred_prob(input_dir, taxon_levels)

  # Set a threshold for filtering max probabilities for each label
  print("Parse thresholds:")
  thre_list = list(map(float, thresh.split(",")))
  print(" ", thre_list)

  print("Read taxa parent-child relationships")
  p_c_file = metadata_dir / "taxa_parent_child.txt"
  p_c_dict, c_p_dict = get_parent_child(p_c_file)
  #print(c_p_dict)

  print("Read taxon name to label file")
  taxon_name_file = metadata_dir / "taxa_name_to_label.txt"
  taxon_name_dict = get_taxa_name_dict(taxon_name_file)
  #print(taxon_name_dict)

  print("Traverse across levels for each instance")
  
  assign_dict = {} # {photoid:[taxon_names, best_prob]}
  debug_dict  = {} # {photoid:[taxon_labels, problist]}

  debug = 0
  # ise class level df to get photoids
  photoids = prob_df_list[0].index
  for photoid in tqdm(photoids): 
    # print("CHECK photoid", photoid)

    # Go through all levels and calculate the joint probability at each level
    best_list = []
    best_prob = 0
    best_prob_list = []
    # best_prob_dict = {}
    for level in range(5):
      # Build pdict to store all info for this photoid. Stored info only contain
      # that specified in the parent-child relationship file.
      # {taxon_leve: {taxon_label: prob}} 
      #print("level:", level)
      labels = prob_df_list[level].columns
      pdict = fill_pdict(photoid, {}, level, labels)
      if debug: print("forward pdict:", pdict)

      # for level deeper than class, back fill info
      if level > 0:
        pdict = back_fill(photoid, pdict, level)
      if debug: print("backfil pdict:", pdict)

      taxon_list, root_prob, probs_list = get_taxon_prob(pdict)
      # print("CHECK 322 best_list:", best_list, 'best_prob:', best_prob)
      if root_prob > best_prob:
        best_list = taxon_list
        best_prob = root_prob
        best_prob_list = probs_list
      # print("CHECK 327 best_list:", best_list, 'best_prob:', best_prob)

    # output this for dubugging purpose
    debug_dict[photoid]  = [best_list, best_prob_list]

    # Kenia Note: Recalculate joint probabilities based on given thresholds for each class
    # check which probs meet the thresholds
    best_list_thresh = [] # taxa levels that meet threshold
    best_prob_list_thresh = [] # probabilities that meet threshold
    for i in range(len(best_prob_list)-1, -1, -1):
      if best_prob_list[i] > thre_list[i]:
        best_prob_list_thresh.append(best_prob_list[i])
        best_list_thresh.append(best_list[i])
      else:
        best_list_thresh.append("")
    
    # further filter probabilities with a second threshold
    # print('CHECK 344', best_prob_list_thresh)
    # print('CHECK 345', best_list_thresh)
    best_prob_list_thresh_2 = []
    for prob in best_prob_list_thresh:
      if prob > 0.2:
        best_prob_list_thresh_2.append(prob)
    
    # recalculate joint probabilities
    if len(best_prob_list_thresh_2)!= 0:
      new_joint_prob = np.prod(best_prob_list_thresh_2)
      #print("new_joint_prob:", new_joint_prob)
      new_root_prob  = np.power(new_joint_prob, 1/len(best_prob_list_thresh_2))
      #print("new_root_prob:", new_root_prob)
    
    # check what the deepest taxa level that met the threshold is
    if len(best_list_thresh)!=0:
      # print('CHECK 352', best_list_thresh)
      for i in range(len(best_list_thresh)):
        if (best_list_thresh[i]!="") & (i==0):
          deepest = "species"
          break
        elif (best_list_thresh[i]!="") & (best_list_thresh[i-1]==""):
          if len(best_list_thresh[i:])==4:
            best_list = [best_list[i] if i<4 else "" for i in range(len(best_list))]
            deepest = "genus"
          elif len(best_list_thresh[i:])==3:
            best_list = [best_list[i] if i<3 else "" for i in range(len(best_list))]
            deepest = "family"
          elif len(best_list_thresh[i:])==2:
            best_list = [best_list[i] if i<2 else "" for i in range(len(best_list))]
            deepest = "order"
          elif len(best_list_thresh[i:])==1:
            best_list = [best_list[i] if i<1 else "" for i in range(len(best_list))]
            deepest = "class"
          break
    else:
      deepest = ""
    #print('CHECK 375', deepest)
    #print('CHECK 376-->', best_list)
    if best_list != []:
      # translate taxon label to taxon name
      best_list_names = []
      for level, taxon_level in enumerate(taxon_levels):
        print(level, taxon_level)
        if best_list[level]!="":
          taxon_label = best_list[level]
          taxon_name  = taxon_name_dict[taxon_level][taxon_label] 
          best_list_names.append(taxon_name)
        else:
          best_list_names.append("")

      # print('CHECK 413', best_list)
      phylum_label = c_p_dict["phylum"][best_list[0]]
      phylum_name  = taxon_name_dict["phylum"][phylum_label]

      # append phyla and kingdom info
      taxa = ["Plantae", phylum_name] + best_list_names

      # replace the last with species epithet
      taxa[-1] = taxa[-1].split(" ")[-1]

      # append taxon rank
      taxa.append(deepest)
      
      # convert prob into %
      # assign_dict[photoid] = [taxa, '{0:.2f}'.format(best_prob*100)]
      # Kenia Note: Assign re-calculated root probability
      if len(best_prob_list_thresh)!= 0:
        assign_dict[photoid] = [taxa, '{0:.2f}'.format(new_root_prob*100)]
      else:
        assign_dict[photoid] = [taxa, '{0:.2f}'.format(0*100)]

      #print(taxa, best_prob)

  print("Generate output")
  input_dir_name = str(input_dir).split("/")[-1]
  
  if args.debug:
    debug_file     = output_dir / f"tmp_debug_{input_dir_name}.csv"
    with open(debug_file, "w") as f:
      f.write("photoid,class,order,family,genus,species,p_C,p_o,p_f,p_g,p_s\n")
      for photoid in debug_dict:
        labels = ','.join(debug_dict[photoid][0])
        probs  = ','.join([str(i) for i in debug_dict[photoid][1]])
        f.write(f"{photoid},{labels},{probs}\n")

  output_file    = output_dir / f"{input_dir_name}.csv"
  with open(output_file, "w") as f:

    method = "ML_plant_ensemble"  # identification_method
    event_id = input_dir.name

    out_csv = csv.writer(f)
    out_csv.writerow(
      ["sampling_event_id", "clip_file_path", "team", "inat_code","preparations",
        "collection_method","identification_method","aiml_name",
        "kingdom","phylum","class","order","family","genus","species",
        "scientific_name","taxon_rank","confidence_percent",
        "verification_method","verification_name","occurrence_remarks",
        "individual_count","organism_quantity","organism_quantity_type",
        "references","notes",])

    for photoid in assign_dict:
      taxa_str = ','.join(assign_dict[photoid][0])
      taxa = taxa_str.split(',')
      gen = taxa[-3]
      spc = taxa[-2]
      txn = taxa[-1]
    
      conf_per = assign_dict[photoid][1]
    
      out_csv.writerow([event_id,input_dir/photoid,"plant_image","NA", "image_segmentation",
                        "MachineObservation",method, method] \
                        + taxa[:-1] \
                        + [gen+' '+spc, txn, conf_per,
                           "NA","NA","NA",
                           "NA","NA","NA",
                           "NA","NA"])

print("Done!")