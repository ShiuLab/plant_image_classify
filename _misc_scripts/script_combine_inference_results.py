'''
Shinhan Shiu
06/01/23
This is modified from script_combine_photoid_infer.py which is for combining
not only the inference results, but also the photoid folder to find out the
true labels.
Combine photo_id, their class_id, and the inference results. This processes the
output log files of efficientnet_inference_plant.py and combine the info on the
logs with true label info in plants_photo-ids.csv.
06/03/23: Bug, the output header label is not corect. 
'''

import argparse, sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-i", "--input_inference_log_dir",
    type=str,
    default="/mnt/research/xprize23/plants_essential/test_data/C201_effnet",
    help="directory with inference log files with probability info",
    required=False,
  )  
  parser.add_argument(
    "-o", "--output_dir",
    type=str,
    default="/mnt/research/xprize23/plants_essential/test_data/C201_effnet",
    help="directory to output combined_infer.csv",
    required=False,
  )  
  parser.add_argument(
    "-e", "--meta_data_dir",
    type=str,
    default="/mnt/research/xprize23/plants_essential/meta_data",
    help="directory with meta data, ",
    required=False,
  )
  return parser.parse_args()

def populate_pdict_with_log(infer_log_dir):
  '''
  Go through inference log files and populate pdict with inference results
  Args:
    infer_log_dir: a directory with inference log files
  Returns:
    pdict: a dictionary with photoid as key, a dictionary as value with:
      {taxon_level: [probs]}. 
    ldict: a dictionary with taxon_level as key, a list of labels as value
  '''

  pdict = {} # for updated information
  ldict = {} # label dictionary: {taxon_level:[labels]}
  # go through inference log files and save info in pdict
  for log_file in infer_log_dir.iterdir():

    # Skill non-log files
    if str(log_file).find("log_inference") == -1:
      continue

    print("", str(log_file).split('/')[-1])
    taxon_level = str(log_file).split('/')[-1].split('_')[2]

    with open(log_file, "r") as f:
      if str(log_file).split("/")[-1].startswith("log_inference"):
        # Get number of classes
        class_labels = f.readline().strip().split(",")[3:]
        ldict[taxon_level] = class_labels

        n_classes   = len(class_labels)
        print(f'  {taxon_level}:{n_classes} classes')
        
        # rid of header
        infer_lines = f.readlines()

        # Get inference results for each photoid
        # file, pred_encoding_label, , best_prob, probs...
        for iline in infer_lines:
          ilist   = iline.strip().split(",")
          photoid = ilist[0] # use the full file name as photoid
          probs   = ilist[3:]
          print(photoid)
          if len(probs) != n_classes:
            print(f"ERR: # probs={len(probs)}, # classes={n_classes}")
            sys.exit(0)

          if photoid not in pdict:
            pdict[photoid] = {taxon_level:probs}
          elif taxon_level not in pdict[photoid]:
            pdict[photoid][taxon_level] = probs

  return pdict, ldict

def check_photoids(pdict):
  '''
  Check if all photoids have inference at all taxon_levels
  Args:
    pdict: a dictionary with photoid as key and taxa info as value
  Returns:
    None
  '''
  ok = 0
  for photoid in tqdm(pdict):
    num_infer = len(pdict[photoid])
    if num_infer != 5:
      print(f"Err: {photoid} have inference for {num_infer} levels")
      print(pdict[photoid])
      sys.exit(0)
    else:
      #print("  have all taxon_levels")
      ok += 1

  print(f"  total:{len(pdict)}, {ok} photoids have all taxon_levels")

def generate_output(pdict, ldict, output_file):
  '''
  Generate output file with photoid, taxa info, and inference results
  Args:
    pdict: a dictionary with photoid as key and taxa info as value
    ldict: a dictionary with taxon_level as key, a list of labels as value
    ouput_file: a file to write output to
  '''
  print("  output to: ", output_file)

  # ensiure the order of taxon levels in header and probs are the same
  taxon_order = ['class', 'order', 'family', 'genus', 'species']
  #print(pdict)
  with open(output_file, "w") as f:

    # Construct prob header string
    # header: photoid, dummy_column_for_label, probs for each class
    #   for each taxon level, prob for each taxan level has column name 
    #   taxonlevel_classlabel
    prob_header = ''
    n_tokens    = 0  # get count to make sure things are correect.
    #print(ldict)
    for taxon_level in taxon_order:
      labels = ldict[taxon_level]
      n_tokens += len(labels)

      tlvl_list = [f'{taxon_level}_{label}' for label in labels]
      prob_header += ','.join(tlvl_list) + ","

    prob_header = prob_header[:-1]  # rid of last comma 
    print(f'  n_tokens:{n_tokens}, header tokens:{len(prob_header.split(","))}')

    f.write(f"photoid,dum1,dum2,dum3,dum4,dum5,{prob_header}\n")

    # Write probabilities
    for photoid in pdict:
      prob_dict = pdict[photoid]

      # dummy column for label
      taxon_label = '0,0,0,0,0'

      prob_str = ''
      for taxon_level in taxon_order:
        probs     = prob_dict[taxon_level]
        prob_str += (','.join(probs) + ",")

      prob_str = prob_str[:-1]  # rid of last comma
      if n_tokens != len(prob_str.split(",")):
        print(f'ERR: {photoid}, n_tokens:{n_tokens}, ' + \
              f'prob_str tokens:{len(prob_str.split(","))}')
        print("QUIT! SHOULD NOT GO TO NEXT STEP!")
        sys.exit(0)      

      f.write(f"{photoid},{taxon_label},{prob_str}\n")

  print("Done")

if __name__ == "__main__":

  # Get arguments
  args          = parse_arguments()
  infer_log_dir = Path(args.input_inference_log_dir)
  output_dir    = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file   = output_dir / "combined_infer.csv"

  print("Populate pdict with inference results")
  pdict, ldict = populate_pdict_with_log(infer_log_dir)

  print("Check photoids")
  check_photoids(pdict)

  print("Generate output")
  generate_output(pdict, ldict, output_file)