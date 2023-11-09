'''
Shinhan Shiu
6/3/23: Discover that the efficientnet model label columns are not treated
  correctly. Get the labels for each taxon label and write them to a file.

Ref:
- https://discuss.pytorch.org/t/how-to-get-the-class-names-to-class-label-mapping/470/3

'''

from pathlib import Path
from torchvision import datasets

base_dir     = Path("/mnt/research/xprize23/")
taxon_levels = ["class","order","family","genus","species"]

output_file  = base_dir / "plants_essential/meta_data/taxon_labels.txt"
with open(output_file, "w") as f:
  for taxon_level in taxon_levels:
    print(taxon_level)
    f.write(f"{taxon_level}\t")
    data_folder = base_dir / f'plants_test/{taxon_level}/{taxon_level}_train'
    print(data_folder)
    dataset = datasets.ImageFolder(data_folder)

    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    f.write(','.join(idx_to_class.values())+'\n')

print("Done!")