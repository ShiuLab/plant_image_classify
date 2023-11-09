'''
Assess how prediction probabilities can be use to maximize accuracy
Shinhan Shiu
05/30/2023

Possible approaches:
1. Max raw prob: the highest probability call is assumed to be correct.
2. Max prob/background: the highest normalized probability (against background)) 
'''

import argparse, sys
import numpy as np
from pathlib import Path

def parse_arguments():
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "-d", "--combined_infer_dir",
    type=str,
    default="/mnt/research/xprize23/plants_test/_infer_original/csv_thres0",
    help="directory with combined_infer.csv",
    required=False,
  )
  parser.add_argument(
    "-f", "--combined_infer_file",
    type=str,
    default="combined_infer.csv",
    required=False,
  )
  
  return parser.parse_args()

def get_classes(taxon_level):
  '''Hard codeded number to make things easier'''
  class_number_dict = {"class"  : 9,
                       "order"  : 59,
                       "family" : 190,
                       "genus"  : 829,
                       "species": 1450,}
  
  num_classes = class_number_dict[taxon_level]
  classes     = [str(i) for i in range(1,num_classes+1)]

  return classes

def get_unique_taxa_combo(comb_infer_file):
  with open(comb_infer_file, "r") as f:
    header = f.readline().strip().split(",")

    # taxon levels
    taxon_levels = header[1:6]
    print("  taxon levels:", taxon_levels)
    
    # go through true labels to ge unique taxa combinations
    # {taxon_level: {taxon_label_list: count}}
    # For example, for classes, the label list is one element long, and for
    # species, the label list is five elements long
    u_taxa_combo = {}
    ci_line = f.readline()
    while ci_line != "":
      ci_fields = ci_line.strip().split(",")

      # {true_taxon_level:true_taxon_label}
      for idx in range(1, 6):
        ttaxon_level  = taxon_levels[idx-1]
        ttaxon_labels = tuple(ci_fields[1:idx+1])  # all labels up to this level
        if ttaxon_level not in u_taxa_combo:
          u_taxa_combo[ttaxon_level] = {ttaxon_labels:1}
        elif ttaxon_labels not in u_taxa_combo[ttaxon_level]:
          u_taxa_combo[ttaxon_level][ttaxon_labels] = 1
        else:
          u_taxa_combo[ttaxon_level][ttaxon_labels] += 1

      ci_line = f.readline()

    print("  unique taxa combinations:")
    for i in u_taxa_combo:
      print("  ", i, len(u_taxa_combo[i]))

    return u_taxa_combo



def get_true_pred_probs(comb_infer_file):

  with open(comb_infer_file, "r") as f:
    header = f.readline().strip().split(",")

    # taxon levels
    taxon_levels = header[1:6]
    print(taxon_levels)

    # indices of predicted labels in a dictionary
    # {header_index: [ptaxon_level, ptaxon_label]}
    idx_pred_dict = {}
    # start from idx=6 because that's when prediction label starts
    for hidx in range(6, len(header)):
      # substract six for the dictoinary beause the probs later start with 0
      idx_pred_dict[hidx-6] = header[hidx].split("_")
    
    # go through true labels and predicted probs for each image
    ci_line = f.readline()

    # {photoid: {taxon_level: [true_label_probs, max_pred_probs]}}
    true_pred_probs = {} 
    c = 0
    while ci_line != "":
      if c%1e3 == 0:
        print(f"  {c/1e3}x1000")
      ci_fields = ci_line.strip().split(",")

      photo_id = ci_fields[0]

      # {true_taxon_level:true_taxon_label}
      tlabel_dict = {}
      for idx in range(1, 6):
        ttaxon_level = taxon_levels[idx-1]
        tlabel_dict[ttaxon_level] = ci_fields[idx]

      # Get prediction probability for each level and label
      plabel_dict = {}
      for col_idx in range(6, len(header)):
        pred_prob  = float(ci_fields[col_idx]) 

        # idx_pred_dict start with idx=0, but col_idx is following header,
        # which starts with idx=6, so need to substract 6
        [ptaxon_level, ptaxon_label] = idx_pred_dict[col_idx-6]
        if ptaxon_level not in plabel_dict:
          plabel_dict[ptaxon_level] = {ptaxon_label:pred_prob}
        elif ptaxon_label not in plabel_dict[ptaxon_level]:
          plabel_dict[ptaxon_level][ptaxon_label] = pred_prob
        else:
          print("ERR: redundant label,", ptaxon_level, ptaxon_label)

      # get the max prob for each level: {[taxon_label, max_prob]}
      max_pred_probs = []
      # prediction probability for true label at each taxon level
      true_label_probs = []
      for taxon_level in taxon_levels:
        # populate max_pred_probs
        labels       = list(plabel_dict[taxon_level].keys())
        probs        = list(plabel_dict[taxon_level].values())
        max_prob_idx = np.argmax(probs) 
        max_prob     = probs[max_prob_idx]
        max_prob_lbl = labels[max_prob_idx]

        max_pred_probs.append([max_prob_lbl, max_prob])

        # populate true_label_probs
        true_label = tlabel_dict[taxon_level]
        try:
          prob = plabel_dict[taxon_level][true_label]
        except KeyError:
          print(f"ERR: {photo_id} at {taxon_level}, no prob for label={true_label}",)
          prob = "NA"

        true_label_probs.append([true_label, prob])
      
        if taxon_level not in true_pred_probs:
          true_pred_probs[photo_id] = \
            {taxon_level: [true_label_probs, max_pred_probs]}
        elif taxon_level not in true_pred_probs[photo_id]:
          true_pred_probs[photo_id][taxon_level] = \
            [true_label_probs, max_pred_probs]
        else:
          print("ERR: redundant probs for", taxon_level, photo_id)

      c += 1
      ci_line = f.readline()

  # generate poutput
  output_file = Path(str(comb_infer_file).replace(".csv", "_max_prob.csv"))
  with open(output_file, "w") as f:  
    # write header
    header_str = ""
    for taxon_level in taxon_levels:
      header_str += ",".join([taxon_level + "_true_label", 
                              taxon_level + "_true_prob", 
                              taxon_level + "_pred_label", 
                              taxon_level + "_pred_prob"])
    f.write(f"photo_id,{header_str}\n")

    for photo_id in true_pred_probs:
      for taxon_idx, taxon_level in enumerate(true_pred_probs[photo_id]):
        # Get [true_label, true_prob] and [pred_label, pred_prob] for this level
        [[true_label, true_prob], [pred_label, pred_prob]] = \
                          true_pred_probs[photo_id][taxon_idx]
        f.write(f',{true_label},{true_prob},{pred_label},{pred_prob}')       


def get_ci_dict(comb_infer_file):
  '''
  Read combined_infer.csv and construct a dictionary
  Args:
    comb_infer_file: a csv file with photoid, taxa info, and inference results
  Returns:
    ci_dict: a dictionary with photoid as key and a list of two dictionary as
      values, {taxon_level:taxon_label} and {taxon_level:{label:prob}}
  '''
  with open(comb_infer_file, "r") as f:
    header = f.readline().strip().split(",")

    # taxon levels
    taxon_levels = header[1:6]
    print(taxon_levels)

    # indices of predicted labels in a dictionary
    # {header_index: [ptaxon_level, ptaxon_label]}
    idx_pred_dict = {}
    for hidx in range(6, len(header)):
      idx_pred_dict[hidx] = header[hidx].split("_")
    
    # go through true labels and predicted probs for each image
    ci_line = f.readline()
    while ci_line != "":
      ci_fields = ci_line.strip().split(",")

      # {true_taxon_level:true_taxon_label}
      tlabel_dict = {}
      for idx in range(1, 6):
        ttaxon_level = taxon_levels[idx-1]
        tlabel_dict[ttaxon_level] = ci_fields[idx]

      # check
      for ttaxon_level in tlabel_dict:
        print(ttaxon_level, tlabel_dict[ttaxon_level])

      # {ptaxon_level:{ptaxon_label:pred_prob}}
      plabel_dict = {}
      for col_idx in range(6, len(header)):
        pred_prob  = float(ci_fields[col_idx]) 
        [ptaxon_level, ptaxon_label] = idx_pred_dict[col_idx]
        if ptaxon_level not in plabel_dict:
          plabel_dict[ptaxon_level] = {ptaxon_label:pred_prob}
        elif ptaxon_label not in plabel_dict[ptaxon_level]:
          plabel_dict[ptaxon_level][ptaxon_label] = pred_prob
        else:
          print("ERR: redundant label,", ptaxon_level, ptaxon_label)

      for ptaxon_level in plabel_dict:
        print(ptaxon_level)
        for ptaxon_label in plabel_dict[ptaxon_level]:
          print("", ptaxon_label, plabel_dict[ptaxon_level][ptaxon_label])
      
      sys.exit(0)

      ci_line = f.readline()

if __name__ == "__main__":
  args = parse_arguments()

  comb_infer_dir = Path(args.combined_infer_dir)

  # read combined_infer.csv
  comb_infer_file = comb_infer_dir / args.combined_infer_file
  
  #u_taxa_combo = get_unique_taxa_combo(comb_infer_file)

  #get_ci_dict(comb_infer_file)

  #call_max_prob(comb_infer_file)
  get_true_pred_probs(comb_infer_file)



########################
'''Deprecated
def call_max_prob(comb_infer_file):

  p_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

  with open(comb_infer_file, "r") as f:
    header = f.readline().strip().split(",")

    # taxon levels
    taxon_levels = header[1:6]
    print(taxon_levels)

    # indices of predicted labels in a dictionary
    # {header_index: [ptaxon_level, ptaxon_label]}
    idx_pred_dict = {}
    # start from idx=6 because that's when prediction label starts
    for hidx in range(6, len(header)):
      # substract six for the dictoinary beause the probs later start with 0
      idx_pred_dict[hidx-6] = header[hidx].split("_")
    
    # go through true labels and predicted probs for each image
    ci_line = f.readline()

    # {taxon_level:{taxon_label:[TP_at_diffferent_p_thre, FP_..., FN_...]}} 
    counts_T_F  = {} 
    while ci_line != "":
      ci_fields = ci_line.strip().split(",")

      # {true_taxon_level:true_taxon_label}
      tlabel_dict = {}
      for idx in range(1, 6):
        ttaxon_level = taxon_levels[idx-1]
        tlabel_dict[ttaxon_level] = ci_fields[idx]

      probs        = np.array([float(prob) for prob in ci_fields[6:]])
      max_prob_idx = np.argmax(probs) 
      max_prob     = probs[max_prob_idx]
      [ptaxon_level, ptaxon_label] = idx_pred_dict[max_prob_idx]

      # True taxon label
      ttaxon_label = tlabel_dict[ptaxon_level]

      TP = [0]*len(p_thresholds)
      FP = [0]*len(p_thresholds)
      FN = [0]*len(p_thresholds)
      for idx, p in enumerate(p_thresholds):
        if max_prob <= p:
          FN[idx] = 1
        else:
          if ptaxon_label == ttaxon_label:
            TP[idx] = 1
          else:
            FP[idx] = 1

      if ptaxon_level not in counts_T_F:
        counts_T_F[ptaxon_level] = {ptaxon_label:[TP, FP, FN]}
      elif ptaxon_label not in counts_T_F[ptaxon_level]:
        counts_T_F[ptaxon_level][ptaxon_label] = [TP, FP, FN]
      else:
        # old values
        [oTP, oFP, oFN] = counts_T_F[ptaxon_level][ptaxon_label]
        # add values
        nTP = [sum(x) for x in zip(TP, oTP)]
        nFP = [sum(x) for x in zip(FP, oFP)]
        nFN = [sum(x) for x in zip(FN, oFN)]
        counts_T_F[ptaxon_level][ptaxon_label] = [nTP, nFP, nFN]
        
      ci_line = f.readline()# update

  output_file = Path(str(comb_infer_file).replace(".csv", "_max_prob.csv"))
  with open(output_file, "w") as f:
    # write header
    f.write(",".join(["taxon_level", "taxon_label"] + \
                      ["TP_p" + str(p) for p in p_thresholds] + \
                      ["FP_p" + str(p) for p in p_thresholds] + \
                      ["FN_p" + str(p) for p in p_thresholds]) + "\n")
    for taxon_level in counts_T_F:
      for taxon_label in counts_T_F[taxon_level]:
        [TP, FP, FN] = counts_T_F[taxon_level][taxon_label]
        f.write(",".join([taxon_level, taxon_label] + \
                         [str(x) for x in TP] + \
                         [str(x) for x in FP] + \
                         [str(x) for x in FN]) + "\n")
'''