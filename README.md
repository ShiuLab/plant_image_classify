# Plant model training and inference

## Clone this repository

```bash
mkdir ~/github
cd ~/github
```

- Clone the repo into the data directory:
```bash
git clone git@github.com:ShiuLab/plant_image_classify.git
cd plant_image_classify
```

## Setup environment

- Create a `conda` environment:
```bash
conda create --name pic python=3.9
conda activate pic
```

- Use the `requirement.txt` file to install required packages:
```bash
pip install -r requirements.txt
```

- Test installed packages: you should see a help message for the training code run_first.py:

```bash
python ./efficient_inference_plant.py -h
```

## Preprocessing

The images should be preprocessed, split into training, validation, and testing folders, and augmented. 

## Number of classes

For the iNaturalist images, here are the number of classes at each taxonomic level (exclude those with only 1 image/class):
* class: 9,
* order: 59,
* family : 190,
* genus : 829,
* species: 1450

## Modeling

### Efficientnet

#### Training

Original code from Model-Soups, modified by Bowen and Longxiu, adapted for efficientnet by Advika, adapted for plant data by Shiu:
* `data_location`: the root directory for the training datasets.
* `validation_data_location`: The root directory for the validation datasets.
* `test_data_location`: The root directory for the test datasets.
* `timm_model`: timm model to finetune (default: tf_efficientnetv2_b0)
* `model_location`: Where to save the models
* `batch_size: default 256
* `workers`: number of workers (default: 4)
* `epochs`: number of epochs (default: 10)
* `lr`: learning rate (default: 1e-3)
* `ckpt_name_prefix`: Prefix for the checkpoint filename (default: ckpt)

Example command line:
```bash
python ./efficient_model_plant.py \
--data_location ./example_data/augmented_family_train/ \
--validation_data_location ./example_data/augmented_family_validation/ \
--test_data_location ./example_data/family_test/ \
--model_location models/ \
--name family_effnetv2-b0_lr1e-3 \
--batch_size 256 \
--epochs 20 \
--lr 1e-3 > log_fam_effnet_bs256_ep20_lr1e-3
```

#### Testing

Original code from Model-Soups, modified by Bowen and Longxiu, adapted for efficientnet by Advika, adapted for plant data by Shiu:

Example command line for testing mode: [Note] the model is too large to put in the repo.
```bash
python ./efficient_inference_plant.py \
-o testing \
-t family \
-i ./example_data/family_test \
-e ./example_data/taxa_encodings/plants_family-encoding.txt \
-m /mnt/research/xprize23/plants_test/models/family_effnetv2-b0_lr1e-3.pth \
-g ./example_data/log_testing 
```

### PCA

The efficientnet models are their own have low performance. To improve upon the model, an ensemble model approach is used. Prior to the ensemble model building, a PCA model for each taxonomic level is generated to reduce the dimension of the inference data. The PCA models are saved in the folder specified by -save.

```bash
labels=(class family genus order species)
path=/mnt/research/xprize23/plants_essential/example_data/tmp_C201/
for label in ${labels[*]}
do
    echo "Apply ${label} fitted PCA models to combined_infer.csv"
    # apply incremental PCA to training data
    model=/mnt/research/xprize23/plants_test/_infer_original/csv_thres0/${label}_PCA_model.pkl
    python /mnt/research/xprize23/inat-plant-train-infer/_misc_code/0_dimension_reduction.py -path $path -Xinfer combined_infer.csv \
        -pcaModel $model -label dum1,dum2,dum3,dum4,dum5 \
        -split n -save /mnt/home/seguraab/Shiu_Lab/Collabs/XPRIZE/combined_infer_${label} \
        -alg ipca -n 200
done
```

### Multinomial regression

The PCA models are used to reduce the dimension of the inference data. The reduced dimension data are then used to do inference using multinomial regression models. The results are stored in the folder specified by -o.

```bash
python ./logisticreg_model_plant.py -h

usage: logisticreg_model_plant.py [-h] -Xtrain XTRAIN -Xval XVAL -label LABEL
                                  -save SAVE -name NAME

Muticlass classification with Logestic regression

optional arguments:
  -h, --help      show this help message and exit

Required Input:
  -Xtrain XTRAIN  path to training feature data file
  -Xval XVAL      path to validation feature data file
  -label LABEL    name of label column in Xtrain dataframe
  -save SAVE      path to save output files (add / at the end)
  -name NAME      save prefex for the modes
```

## Inference details

The taxa inference involves six steps detailed below.

### Relevant directories

- meta_data
  - `plant_[level]_encoding.txt`: Taxon level encoding files
  - `plant_photo-ids.csv`: mainly for extracting parent/child relations
  - `taxa.csv`: Taxa encoding to taxa name file
  - `taxa_name_to_label.txt`: taxa name to classification model label mapping
  - `taxa_parent_child.txt`: mapping of parent taxa label to child taxa tabels
  - `taxon_labels.txt`: taxon level and associated labels in models
- models
  - `effnet`: 5 Effnetv2-b0 models with learning rate=1e-3 each for a different taxonomic level.
  - `pca`: 5 PCA models, each for a different taxonomic level.
  - `logistic`: 5 Multinomial regression models, each for a different taxonomic level.
- example_data
  - some images for testing purpose

### Procedures

#### Step 1: tile images

Claire has incorporated Daniel's code `image_patches.py` in the`inat-data_aug` repo into the pipeline to generate tile images. 

The run below assume the cloned `inat-plant-train-infer` repo is in your home directory.

#### Step 2: inference using efficientnet models

In the example below, images in original are used to do inference. The results are stored in 5 outputs: `log_inference_[taxon_leve]_effnet.pth` in the folder specified by -o.

```bash
#SBATCH --reservation xprize
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name infer_original
#SBATCH --partition=nal-[000-010]
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%x-%j.out

module purge
conda activate xprize

cd /mnt/research/xprize23/plants_essential

python ~/github/inat-plant-train-infer/efficient_inference_plant.py \
  -i example_data/C201 \
  -o example_data/tmp_C201 \
  -e meta_data \
  -L meta_data/taxon_labels.txt \
  -m models/effnet \
  -d inference \
  -p 0.1 
```

#### Step 3: process effnet results

Combine effnet results to generate output `combined_infer.csv`

```bash
python ~/github/inat-plant-train-infer/_misc_scripts/script_combine_inference_results.py \
  -i example_data/tmp_C201 \
  -o example_data/tmp_C201
```

#### Step 4: dimension reduction

Create reduced dimension data, output `[taxon_level]_combine_infer_PCA.csv`

```bash
python ~/github/inat-plant-train-infer/_misc_scripts/0_dimension_reduction.py \
-path example_data/tmp_C201 \
-Xinfer combined_infer.csv \
-label dum1,dum2,dum3,dum4,dum5 \
-split n \
-n 200 \
-pcaModelDir /mnt/research/xprize23/plants_essential/models/pca \
-save combined_infer 
```

#### Step 5: inference using multinomial regression

Use reduced dimension data to do inference using multinomial regression models. The results are stored: `log_inference_[taxon_leve]_logisticreg.csv`.

```bash
python ~/github/inat-plant-train-infer/logisticreg_inference_plant.py \
  -i example_data/tmp_C201/ \
  -o example_data/tmp_C201 \
  -m models/logistic/
```

#### Step 6: generate final inference

Output `[input_folder_name].csv` C201.csv and in this case it is `tmp_C201.csv` located `inside example_data/tmp_C201`. This file is comma delimited and has predicted taxa with 1 joint probability with the following field:

`photoid,kingdom,phyla,class,order,family,genus,species_epithet,taxon_rank,identification_method,confidence_percent`

In this step, prediction score (or proability) thresholds are used to determine if we are confidence about predictions at different levels. These threshoulds were determined with a false positive rate of 0.5%. These thresholds have been set as default in the code.

```bash
python ~/github/inat-plant-train-infer/_misc_scripts/script_final_output_v2.py \
 -i example_data/tmp_C201/ \
 -o example_data/tmp_C201 \
 -m meta_data \
```

### Inference using example data from competition

```bash
cd /mnt/research/xprize23/plants_essential/

python ~/github/inat-plant-train-infer/efficient_inference_plant.py \
  -i example_data/D501 \
  -o example_data/tmp_D501 \
  -e meta_data \
  -L meta_data/taxon_labels.txt \
  -m models/effnet \
  -d inference \
  -p 0.1 

python ~/github/inat-plant-train-infer/_misc_scripts/script_combine_inference_results.py \
  -i example_data/tmp_D501 \
  -o example_data/tmp_D501

python ~/github/inat-plant-train-infer/_misc_scripts/0_dimension_reduction.py \
-path example_data/tmp_D501 \
-Xinfer combined_infer.csv \
-label dum1,dum2,dum3,dum4,dum5 \
-split n \
-n 200 \
-pcaModelDir /mnt/research/xprize23/plants_essential/models/pca \
-save combined_infer

python ~/github/inat-plant-train-infer/logisticreg_inference_plant.py \
  -i example_data/tmp_D501/ \
  -o example_data/tmp_D501 \
  -m models/logistic/

python ~/github/inat-plant-train-infer/_misc_scripts/script_final_output_v2.py \
 -i example_data/tmp_D501/ \
 -o example_data/tmp_D501 \
 -m meta_data \
```
