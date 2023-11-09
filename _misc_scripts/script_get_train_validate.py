'''
Split combined photoids, taxon_label, and probs into train and validate based
on the original split
Shinhan Shiu
05/29/2023
'''

import sys
from pathlib import Path
from tqdm import tqdm


def get_photoid_dict(photo_dir):
  '''Get a dictionary of photoid and taxon label
  Args:
    photo_dir: Path object to the directory containing photos
  Return:
    photoid_dict: a dictionary with keys as photo ids and values as taxon labels
  '''
  label_subdirs = photo_dir.iterdir()
  
  photoid_dict = {}
  for label_subdir in label_subdirs:
    label = str(label_subdir).split("/")[-1]
    photo_files = list(label_subdir.iterdir())
    for pfile in photo_files:
      if pfile.suffix == ".jpg":
        photoid = str(pfile).split("/")[-1].split(".")[0]
        photoid_dict[photoid] = label

  return photoid_dict

def get_comb_infer_dict(combined_infer_csv):
  '''Get a dictionary of photoid as key and line as value
  Args:
    combined_infer_csv: Path object to the combined_infer.csv file
  Return:
    header (str): header of the combiend infer csv file
    comb_infer_dict: a dictionary with keys as photo ids and line as value
  '''
  with open(combined_infer_csv, "r") as f:
    header               = f.readline()
    combined_infer_lines = f.readlines()

  comb_infer_dict = {}
  for line in combined_infer_lines:

    # The original photo id has _original, rid of it. But this may break things
    # in other situation
    photoid = line.split(",")[0].split("_original")[0]
    comb_infer_dict[photoid] = line

  return header, comb_infer_dict

def write_output_csv(header, comb_infer_dict, photoid_dict, taxon_level,
                     split):
  '''Write output csv file
  Args:
    header: header of the combiend infer csv file
    combined_infer_dict: a dictionary with keys as photo ids and line as value
    photoid_dict: train_dict or valid_dict 
    taxon_level: species, genus, family, order, or class
  '''

  # index in combined_infer.csv
  taxon_idx = {"class": 1, "order": 2, "family": 3, "genus": 4, "species": 5}
  with open(out_dir / f'{taxon_level}_{split}.csv', 'w') as f:

    # write header but rid of irrelvant columns
    llist = header.split(",")
    new   = [llist[0], llist[taxon_idx[taxon_level]]] + llist[6:]
    f.write(','.join(new))
    
    for photoid in photoid_dict:
      if photoid in comb_infer_dict:
        # process line to contain only the intended taxon level
        llist = comb_infer_dict[photoid].split(",")
        new   = [llist[0], llist[taxon_idx[taxon_level]]] + llist[6:]
        f.write(','.join(new))


def iterate_taxon_levels(work_dir, out_dir):

  print("Reading combined_infer.csv")
  header, comb_infer_dict = get_comb_infer_dict(combined_infer_csv)

  print("Constructing photoid dictionary for each taxon level")
  taxon_levels = ["class", "order", "family", "genus", "species"]
  taxon_dict   = {}
  for taxon_level in taxon_levels:
    print("  ", taxon_level")
    train_dir = work_dir / taxon_level / f"{taxon_level}_train"
    valid_dir = work_dir / taxon_level / f"{taxon_level}_validation"

    # train_dict = {}
    train_dict = get_photoid_dict(train_dir)
    valid_dict = get_photoid_dict(valid_dir)

    write_output_csv(header, comb_infer_dict, train_dict, taxon_level, "train")
    write_output_csv(header, comb_infer_dict, valid_dict, taxon_level, "valid")
  

######
work_dir           = Path("/mnt/research/xprize23/plants_test")
out_dir            = work_dir / '_infer_original/csv_thres0'
combined_infer_csv = out_dir / "combined_infer.csv"

iterate_taxon_levels(work_dir, out_dir)

print("Done")