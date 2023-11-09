'''
Shin-Han Shiu
6/1/2023
Convert taxa to parent-child relationship
'''

import sys
from pathlib import Path
from tqdm import tqdm

meta_data_dir = Path("/mnt/research/xprize23/plants_essential/meta_data/")

print("Read encoding files")
meta_data_files = meta_data_dir.iterdir()

# {taxon_level:{encoding:taxon_label}}
enc_dict = {}
for meta_data_file in meta_data_files:
  if str(meta_data_file).find('-encoding.txt') != -1:
    taxon_level = str(meta_data_file).split('/')[-1].split('-')[0].split('_')[1]
    print(" ", taxon_level)
    with open(meta_data_file, "r") as f:
      # {encoding:taxon_label}
      enc_dict[taxon_level] = \
          {l.strip().split(" ")[1]:l.split(" ")[0] for l in f.readlines()}

#print([i for i in enc_dict.keys()])

print("Process taxa.csv")
taxa_csv = meta_data_dir / "taxa.csv"
#taxa_csv = meta_data_dir / "test_taxa.csv"
target_levels = ["kingdom","phylum","class","order","family","genus","species"]
name_to_label = {} # {taxon_level: {taxon_name:taxon_label}}]}
parent_child  = {} # {taxon_level: {parent_taxon_label:[child_taxon_label]}}

c = 0
with open(taxa_csv, "r") as f:
  lines = f.readlines()[1:]
  for line in lines:
    fields      = line.strip().split("\t")
    taxon_level = fields[3]

    if taxon_level in target_levels:
      taxon_id     = fields[0] 
      ancestry     = fields[1].split("/")
      taxon_name   = fields[4]

      # There are situations wehre taxon_id is not in encoding file because
      # they are not plants
      try:
        taxon_label  = enc_dict[taxon_level][taxon_id]
        if taxon_level not in name_to_label:
          name_to_label[taxon_level] = {taxon_name:taxon_label}
        elif taxon_name not in name_to_label[taxon_level]:
          name_to_label[taxon_level][taxon_name] = taxon_label

        if taxon_level in target_levels[1:]:  
          parent_id    = ancestry[-1]
          parent_level = target_levels[target_levels.index(taxon_level)-1]
          parent_label = enc_dict[parent_level][parent_id]
        
          #print(\
          #  f"child : level ={taxon_level}, id={taxon_id}, label={taxon_label}\n"+\
          #  f"parent: level ={parent_level}, id={parent_id}, label={parent_label}\n")

          if parent_level not in parent_child:
            parent_child[parent_level] = {parent_label:[taxon_label]}
          elif parent_label not in parent_child[parent_level]:
            parent_child[parent_level][parent_label] = [taxon_label]
          else:
            parent_child[parent_level][parent_label].append(taxon_label)
        c += 1
      except KeyError as e:
        pass

print(f"  total:{len(lines)}, in encoding:{c}")

print("Generate outputs")
# {taxon_level: {taxon_name:taxon_label}}]}
name_to_label_file = meta_data_dir / "taxa_name_to_label.txt"
print(" ", name_to_label_file)
with open(name_to_label_file, "w") as f:
  # write header
  f.write("taxon_level\ttaxon_name\ttaxon_label\n")
  for taxon_level in name_to_label:
    print("  ", taxon_level)
    for taxon_name in name_to_label[taxon_level]:
      taxon_label = name_to_label[taxon_level][taxon_name]
      f.write(f"{taxon_level}\t{taxon_name}\t{taxon_label}\n")

parent_child_file = meta_data_dir / "taxa_parent_child.txt"
print(" ", parent_child_file)
with open(parent_child_file, "w") as f:
  # write header
  f.write("taxon_level\tlabel\tchild_labels\n")
  for taxon_level in parent_child:
    print("  ", taxon_level)
    for parent_label in parent_child[taxon_level]:
      child_labels = parent_child[taxon_level][parent_label]
      f.write(f"{taxon_level}\t{parent_label}\t{','.join(child_labels)}\n")