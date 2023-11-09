# evaluate multinomial logistic regression model
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import datatable as dt
import pandas as pd
from sklearn.metrics import accuracy_score
import argparse
import time
import pickle

# Argument parser
parser = argparse.ArgumentParser(
    description="Muticlass classification with Logestic regression")

# Required input
req_group = parser.add_argument_group(title="Required Input")
req_group.add_argument(
    "-Xtrain", help="path to training feature data file", required=True)
req_group.add_argument(
    "-Xval", help="path to validation feature data file", required=True)
req_group.add_argument(
    "-label", help="name of label column in Xtrain dataframe", required=True)
req_group.add_argument(
    "-save", help="path to save output files (add / at the end)", required=True)
req_group.add_argument(
    "-name", help="save prefex for the modes", required=True)
# Optional input
#opt_group = parser.add_argument_group(title="Optional Input")
#opt_group.add_argument(
#    "-fold", type=int, help="k number of cross-validation folds", default=5)
#opt_group.add_argument(
#    "-n", type=int, help="number of cross-validation repetitions", default=10)
args = parser.parse_args() # Read arguments
#Output_results_file
out = open(f"{args.save}{args.name}_RESULTS.txt","w")
out.write(f"This is a logestic regression model for multi class classification\n")

# Read in training and validation data
X_train = dt.fread(args.Xtrain) # training data
X_train = X_train.to_pandas()
Y_train = X_train[args.label] # labels for training
X_train = X_train.drop([args.label], axis=1) # drop labels from feature table
X_train.set_index(X_train.columns[0], inplace=True)
X_val = dt.fread(args.Xval) # validation data
X_val = X_val.to_pandas()
Y_val = X_val[args.label] # labels for validation
X_val = X_val.drop([args.label], axis=1) # drop labels from feature table
X_val.set_index(X_val.columns[0], inplace=True)


#training the model
start = time.time()
# define the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# fit the model on the whole dataset
model.fit(X_train, Y_train)
#getting cross val scores
n_scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', n_jobs=-1)
# report the model performance
out.write('Mean Accuracy of the training: %.3f (%.3f) \n' % (mean(n_scores), std(n_scores)))
run_time = time.time() - start
print("Training Run Time: %f" % (run_time))
#predict labels on test dataset
y_pred_test = model.predict(X_val)
score = accuracy_score(Y_val, y_pred_test)
out.write(f'Accuracy on the testing data: {score}\n')
#outputting the prediction probabilities
pred_probs = pd.DataFrame(model.predict_proba(X_val))
pred_probs.columns = model.classes_
pred_probs.index = X_val.index
pred_probs.to_csv(f"{args.save}{args.name}_prediction_probabilities.csv")
#
# Save the fitted model to a file
filename = f"{args.save}{args.name}_model.save"
pickle.dump(model, open(filename, 'wb'))
out.close()
