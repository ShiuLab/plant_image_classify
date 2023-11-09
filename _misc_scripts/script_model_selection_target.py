'''
Shin-Han Shiu, 5/27/2023
[copilot] This script is used to select the best model for the target dataset.

Target models are selected based on:
1. https://paperswithcode.com/sota/image-classification-on-imagenet
2. Model availability in timm
'''

import timm

timm_models = timm.list_models()

# vit include maxvit
target = "tf_efficientnet"

targeted_models = []
for model in timm_models:
  if target in model:
      targeted_models.append(model)

print("# targeted_models:", len(targeted_models))

# save to file
with open("targeted_models.txt", "w") as f:
  for model in targeted_models:
      f.write(model + "\n")

print("Write shell script")

header = \
'''#!/bin/bash --login
#SBATCH --reservation xprize
#SBATCH --time=48:00:00     
#SBATCH --nodes=1           
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=32G   
#SBATCH --job-name fam_efficientnet_model_selection
#SBATCH --partition=nal-[000-010]
#SBATCH --gres=gpu:4 
'''

cmd = "python /mnt/research/xprize23/inat-plant-train-infer/efficient_model_SHIU_COPY.py"

with open("run_fam_effnet_model_select.sh", "w") as f:
  f.write(header)
  f.write("\n")
  f.write("module purge\n")
  f.write("conda activate xprize\n")
  f.write("cd /mnt/research/xprize23/plants_test/family/model_selection\n") 
  for model in targeted_models:
    f.write(f"{cmd} --timm_model {model} > log_family_{model}\n")
