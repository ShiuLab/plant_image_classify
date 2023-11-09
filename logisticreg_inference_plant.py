'''
Thilanka Ranaweera
5/31/2023
this is a code to implement trained model on new testing data
Inputs: 1. Saved model
  2. testing data with lable information 
  3. save name for final file
  4. path to save files to
Output: 1. Results file containing accuracy
  2. Prediction probabilities for each class

6/1/2023: Modified by Shiu 
- Allow looping through multiple models
- Rid of warning: 
  - X has feature names, but LogisticRegression was fitted without feature names
'''

import joblib
import argparse
import pandas as pd
import datatable as dt
from pathlib import Path
import os

def parse_arguments():
  parser = argparse.ArgumentParser(
  description="Testing Muticlass classification with Logestic regression")
  
  parser.add_argument(
    "-i", "--input_dir",
    type=str,
    help="input dir with csv for inference, from 0_dimension_reduction.py",
    required=True,
  )
  parser.add_argument(
    "-o", "--output_dir",
    type=str,
    help="directory to generate outputs",
    required=True,
  )
  parser.add_argument(
    "-m", "--model_dir",
    type=str,
    help="directory containing logistic regression models for 5 taxon levels",
    required=True,
  )
  parser.add_argument(
    "-L", "--label_file",
    type=str,
    default="/mnt/research/xprize23/plants_essential/meta_data/taxon_labels.txt",
    help="file with label information",
    required=False
  )
  return parser.parse_args()

if __name__ == "__main__":

  args       = parse_arguments() # Read arguments
  input_dir  = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  model_dir = Path(args.model_dir)

  label_file = Path(args.label_file)
  label_dict = {} # {taxon_level:labels} <--- int
  with open(label_file) as f:
    lines = f.readlines()
    for line in lines:
      [taxon_level, labels] = line.strip().split("\t")
      labels = [int(label) for label in labels.split(",")]  
      labels.sort()
      label_dict[taxon_level] = labels

  #print(label_dict)

  # Loop through all models
  print("Make predictions")
  for model_file in model_dir.iterdir():
    taxon_level = str(model_file).split('/')[-1].split('_')[0]
    print(f"  {taxon_level}")

    print("   load mode:",str(model_file).split('/')[-1])
    model = joblib.load(model_file)

    print("   read data",f"{taxon_level}_combined_infer_PCA.csv")
    pca_csv = input_dir / f"{taxon_level}_combined_infer_PCA.csv"
    X_test = dt.fread(pca_csv, header=True) 
    X_test = X_test.to_pandas()
    print(X_test)
    X_test.columns = ["photoid"] + [str(i) for i in range(0, X_test.shape[1]-1)]
    #X_test.drop(X_test.index[0], inplace=True)

    X_test.set_index(X_test.columns[0], inplace=True)
    #outputting the prediction probabilities
    output_file = os.path.join(output_dir,f"log_inference_{taxon_level}_logisticreg.csv")

    # Convert dataframe to numpy array to prevent warning:
    # UserWarning: X has feature names, but LogisticRegression was fitted 
    #   without feature names
    # See: https://stackoverflow.com/questions/74232045/userwarning-x-does-not-have-valid-feature-names-but-linearregression-was-fitte
    probs  = model.predict_proba(X_test.to_numpy())
    df_probs = pd.DataFrame(probs)

    df_probs.columns = model.classes_
    #print(model.classes_)
    #print(df_probs.shape)
    #print(df_probs.columns)
    df_probs.index = X_test.index
    df_probs.to_csv(output_file)
  
  print("Done!")
