'''
Shinhan Shiu
05/29/2023
Combine photo_id, their class_id, and the inference results. This processes the
output log files of efficientnet_inference_plant.py and combine the info on the
logs with true label info in plants_photo-ids.csv.

Example cmd line:
python _misc_scripts/script_combine_photoid_infer.py \
  -m meta_data \
  -l /mnt/research/xprize23/plants_test/_infer_original/_log_thre0.1/ \
  -o  /mnt/research/xprize23/plants_test/_infer_original/csv_thres0.1/ \
  -D "_"

'''

import argparse, sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_arguments():
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "-p", "--photoid_dir",
    type=str,
    default="/mnt/research/xprize23/plants_essential/meta_data",
    help="directory with file with photoid and taxa info, ",
    required=False,
  )
  parser.add_argument(
    "-l", "--infer_log_dir",
    type=str,
    default="/mnt/research/xprize23/plants_test/_infer_original/_log/",
    help="directory with inference log files with probability info",
    required=False,
  )  
  parser.add_argument(
    "-o", "--output_dir",
    type=str,
    default="/mnt/research/xprize23/plants_test/_infer_original/",
    help="directory to output combined_infer.csv",
    required=False,
  )  
  parser.add_argument(
    "-D", "--photoid_delimiter",
    type=str,
    default='',
    help="the delimiter for separating photoid from file name, this is \
      particularly important for tiled (XXXX_YYY-ZZZ.jpg) or augmented images \
      (XXXX-augYY.jpg) where extensions are addded to the original name, \
      this is different from '.' which will be used already remove .jpg; \
      (default: '')",
    required=False,
  )  
  parser.add_argument(
    "-m", "--meta_data_dir",
    type=str,
    default='/mnt/research/xprize23/plants_essential/meta_data',
    help="meta data directory wtih ",
    required=False,
  )  
  return parser.parse_args()

def get_photoid_dict(photoid_file):
  '''
  Read photoid_file and construct a dictionary
  Args:
    photoid_file: a csv file with photoid and taxa info
  Returns:
    pdict: a dictionary with photoid as key and taxa info as value,
      {photoid: [taxon_label, {taxon_level: [probs]}]
  '''
  with open(photoid_file, "r") as f:
    # skip header: 
    photoid_lines = f.readlines()[1:]

  # go through photoids and save info in a dictionary
  pdict = {} 
  for pline in photoid_lines:
    pfields     = pline.strip().split(",")
    photoid     = pfields[1]
    taxon_label = pfields[2:9]
    pdict[photoid] = [taxon_label, {}]

  return pdict

def get_class_dict(taxon_label_file):

  class_dict = {} # {taxon_level:[taxon_labels]}
  with open(taxon_label_file, "r") as f:
    lines = f.readlines()
    for line in lines:
      taxon_level = line.split("\t")[0]
      taxon_labels = line.strip().split("\t")[1].split(",")
      class_dict[taxon_level] = taxon_labels

  return class_dict

def populate_pdict_with_log(pdict, cdict, infer_log_dir):
  '''
  Go through inference log files and populate pdict with inference results
  Args:
    pdict: a dictionary with photoid as key and taxa info as a dictionary,
      {photoid: [taxa_info, {taxon_level: [probs]}]
    cdict: {taxon_level:[taxon_labels]}
    infer_log_dir: a directory with inference log files
  Returns:
    pdict_upd: updated pdict with inference results
  '''
 
  pdict_upd = {} # for updated information
  # go through inference log files and save info in pdict
  for log_file in infer_log_dir.iterdir():
    taxon_level = str(log_file).split('/')[-1].split('_')[2]
    n_classes   = len(cdict[taxon_level])
    print(f'  {taxon_level}:{n_classes} classes')

    with open(log_file, "r") as f:
      if str(log_file).split("/")[-1].startswith("log_inference"):
        print("",log_file)
        # rid of header
        infer_lines = f.readlines()[1:]

        # Get inference results for each photoid
        # file, pred_encoding_label, , best_prob, probs...
        for iline in infer_lines:
          ilist   = iline.strip().split(",")
          photoid = ilist[0].split('.')[0]
          probs   = ilist[3:]

          if len(probs) != n_classes:
            print(f"ERR: # probs={len(probs)}, # classes={n_classes}")
            sys.exit(0)

          # photoid XXXX is extracted from file name which can be:
          #   XXXX_original.jpg: original image
          #   XXXX-augYY.jpg: augmented image
          #   XXXX_YYY-ZZZ.jpg: tiled image
          photoid_original = photoid
          if photoid_delimiter != '':
            photoid_original = photoid.split(photoid_delimiter)[0]

          if photoid_original in pdict:
            # copy all info to updated dictionary
            pdict_upd[photoid] = pdict[photoid_original]
            pdict_upd[photoid][1][taxon_level] = probs
          else: # XXXX.jpg                                                      
            print("ERR: photoid not found in photoid_file: ", photoid)

  return pdict_upd

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
    num_infer = len(pdict[photoid][1])
    if num_infer != 5:
      print(f"Err: {photoid} have inference for {num_infer} levels")
      print(pdict[photoid])
      sys.exit(0)
    else:
      #print("  have all taxon_levels")
      ok += 1

  print(f"  total:{len(pdict)}, {ok} photoids have all taxon_levels")

def generate_output(pdict, cdict, output_file):
  '''
  Generate output file with photoid, taxa info, and inference results
  Args:
    pdict: a dictionary with photoid as key and taxa info as value
    cdict: {taxon_level:[taxon_labels]}
    ouput_file: a file to write output to
  '''
  print("  output to: ", output_file)

  # ensiure the order of taxon levels in header and probs are the same
  taxon_order = ['class', 'order', 'family', 'genus', 'species']

  with open(output_file, "w") as f:

    # Construct prob header string
    # header: photoid, class, order, family, genus, species, probs for each class
    #   for each taxon level
    # prob for each taxan level has column name taxonlevel_classlabel
    prob_header = ''
    n_tokens    = 0  # get count to make sure things are correect.
    for taxon_level in taxon_order:
      labels    = cdict[taxon_level]
      n_classes = len(cdict[taxon_level])
      n_tokens += n_classes

      tlvl_list = [f'{taxon_level}_{label}' for label in labels]
      prob_header += ','.join(tlvl_list) + ","

    prob_header = prob_header[:-1]  # rid of last comma 
    print(f'  n_tokens:{n_tokens}, header tokens:{len(prob_header.split(","))}')

    f.write(f"photoid,class,order,family,genus,species,{prob_header}\n")

    # Write probabilities
    for photoid in pdict:
      prob_dict = pdict[photoid][1]

      # Rid of the 1st two label: kingdom and phylum
      taxon_label = ','.join(pdict[photoid][0][2:])

      prob_str = ''
      for taxon_level in taxon_order:
        probs     = prob_dict[taxon_level]
        prob_str += (','.join(probs) + ",")

      prob_str = prob_str[:-1]  # rid of last comma
      if n_tokens != len(prob_str.split(",")):
        print(f'ERR: {photoid}, n_tokens:{n_tokens}, ' + \
              f'prob_str tokens:{len(prob_str.split(","))}')
        sys.exit(0)      

      f.write(f"{photoid},{taxon_label},{prob_str}\n")

  print("Done")

if __name__ == "__main__":

  # Get arguments
  args          = parse_arguments()
  photoid_dir   = Path(args.photoid_dir)
  infer_log_dir = Path(args.infer_log_dir)
  output_dir    = Path(args.output_dir)
  meta_data_dir = Path(args.meta_data_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file   = output_dir / "combined_infer.csv"
  photoid_delimiter = args.photoid_delimiter

  # ['',id,kingdom,phylum,class,order,family,genus,species,filename]
  # The above contains taxa_ids instead of taxa_names.
  photoid_file  = photoid_dir / 'plants_photo-ids.csv'

  print("Get class dictionary")
  taxon_labels_file = meta_data_dir / 'taxon_labels.txt'
  cdict = get_class_dict(taxon_labels_file)

  print("Reading photoid_file and construct dictionary")
  pdict = get_photoid_dict(photoid_file)

  print("Populate pdict with inference results")
  pdict = populate_pdict_with_log(pdict, cdict, infer_log_dir)

  print("Check photoids")
  check_photoids(pdict)

  print("Generate output")
  generate_output(pdict, cdict, output_file)