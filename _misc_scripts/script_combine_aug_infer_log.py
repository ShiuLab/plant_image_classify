
'''
Shin-Han Shiu
5/21/2023
- For the folder with augmented images, the log files of inferenece results are 
for different taxon_levels and each classes. To combine these log files with 
the true label file with script_combine_photoid_infer.py, these log files are 
consolidated so one file is created for each taxonomic level. 
- This also potentially will be used for inference data in multiple directories

Example cmd line:

'''

import argparse, sys
from pathlib import Path
from tqdm import tqdm

def parse_arguments():
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "-l", "--log_file_dir",
    type=str,
    default="/mnt/research/xprize23/plants_test/_infer_aug/_log_thre0_sp_train",
    help="directory with log files",
    required=False,
  )

  parser.add_argument(
    "-o", "--output_dir",
    type=str,
    default="/mnt/research/xprize23/plants_test/_infer_aug/_log_thre0_sp_train_combined",
    help="directory to output combined log files",
    required=False,
  )

  return parser.parse_args()

if __name__ == '__main__':

  args = parse_arguments()

  log_file_dir = Path(args.log_file_dir)
  output_dir   = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # {out_filename:[log_file_paths]}
  print("Iterate through directories to build a dictionary")
  logs_to_combine = {}
  for log_file in log_file_dir.iterdir():
    taxon_level  = log_file.stem.split('_')[2]
    if taxon_level not in logs_to_combine:
      logs_to_combine[taxon_level] = [log_file]
    else:
      logs_to_combine[taxon_level].append(log_file)

  print("Generate output")
  for taxon_level in logs_to_combine:
    print(" ",taxon_level)
    output_file = output_dir / \
      f"log_inference_{taxon_level}_effnet_bs256_ep20_lr1e-3.txt"

    with open(output_file, "w") as f:
      # write header
      f.write("file,pred,conf,probs\n")
      for log_file in tqdm(logs_to_combine[taxon_level]):
        with open(log_file, "r") as lf:
          # alredy have end-of-line character
          f.write("".join(lf.readlines()[1:]))

