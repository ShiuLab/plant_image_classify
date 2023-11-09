'''
The other version using taxa.csv and I found that there are some taxa missing.
So creat this and use plants_photo-ids.csv instead.
'''

from pathlib import Path

infile = Path("/mnt/research/xprize23/plants_essential/meta_data/plants_photo-ids.csv")
outfile = Path("/mnt/research/xprize23/plants_essential/meta_data/taxa_parent_child.txt")


# {taxon_level:{parent_label:child_label}}
p_c_dict = {}

with open(infile) as f:
  header = f.readline()

  taxa_levels = header.split(",")[2:9]
  print(taxa_levels)
  
  lines = f.readlines()[1:]
  for line in lines:
    fields = line.strip().split(",")
    taxa   = fields[2:9]
    level = 0
    for taxon1, taxon2 in zip(taxa[:-1], taxa[1:]):
      taxon_level = taxa_levels[level]
      if taxon_level not in p_c_dict:
        p_c_dict[taxon_level] = {taxon1:{taxon2:1}}
      elif taxon1 not in p_c_dict[taxon_level]:
        p_c_dict[taxon_level][taxon1] = {taxon2:1}
      elif taxon2 not in p_c_dict[taxon_level][taxon1]:
        p_c_dict[taxon_level][taxon1][taxon2] = 1
      else:
        p_c_dict[taxon_level][taxon1][taxon2] += 1
      level += 1

with open(outfile, "w") as f:
  f.write("taxon_level\tlabel\tchild_labels\n")

  for taxon_level in p_c_dict:
    for parent in p_c_dict[taxon_level]:
      children = ','.join(list(p_c_dict[taxon_level][parent].keys()))

      f.write(f"{taxon_level}\t{parent}\t{children}\n")


