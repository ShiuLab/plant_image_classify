'''Testing other timm models with fastai
Shin-Han Shiu, 5/27/2023
https://timm.fast.ai/
'''

import argparse
from fastai.vision.all import *

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--img_base_dir",
      type=str,
      default=os.path.expanduser('/mnt/research/xprize23/plants_test/family'),
      help="Path of the root directory with images, " +\
           "(default: /mnt/research/xprize23/plants_test/family))",
  )
  parser.add_argument(
      "--train_dir_name",
      type=str,
      default=os.path.expanduser('augmented_family_train'),
      help="Training directory name (default: augmented_family_train)",
  )
  parser.add_argument(
      "--validation_dir_name",
      type=str,
      default=os.path.expanduser('augmented_family_validation'),
      help="Validation directory name (defaut: augmented_family_validation)",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=256,
      help="Validation directory name (default: 256)",
  )

  return parser.parse_args()

if __name__ == "__main__":
  # Get arguments
  args           = parse_arguments()
  img_base_dir   = args.img_base_dir
  train_dir_name = args.train_dir_name
  valid_dir_name = args.validation_dir_name
  batch_size     = args.batch_size

  dls = ImageDataLoaders.from_folder(
    img_base_dir, train=train_dir_name, valid=valid_dir_name, bs=batch_size)
  dls.show_batch()