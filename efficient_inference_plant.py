"""Module to run the insect family classifier model on either a single test 
image or a batch of images. If GPU is available, model uses GPU (inference is 
3x faster than CPU). Model also works on CPU.

5/28/23: [Shinhan Shiu] Modified for plant images
6/3/23: Modified to allow labels to be in the header line
"""

import argparse, os, sys, warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from time import time

import torch
from torchvision import datasets, transforms
#from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

import timm
from timm.loss import LabelSmoothingCrossEntropy

warnings.filterwarnings("ignore")

class ImageFolderWithPaths(datasets.ImageFolder):
  """Custom dataset that includes image file paths. Extends
  torchvision.datasets.ImageFolder, from:
  https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
  """

  # override the __getitem__ method. this is the method that dataloader calls
  def __getitem__(self, index):
    # this is what ImageFolder normally returns 
    original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
    # the image file path
    path = self.imgs[index][0]
    # make a new tuple that includes original and the path
    tuple_with_path = (original_tuple + (path,))
    return tuple_with_path

## arguments that can be changed
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-t", "--taxon_level",
    type=str,
    default="family",
    help="taxon level for testing mode: class, order, family, genus, species",
    required=False,
  )
  parser.add_argument(
    "-i", "--image_path",
    type = str,
    help = "The root directory for input datasets to test model performance \
      in testing mode, or the path to image/folder of images to predict in \
      inference mode",
    required=True,
  )
  parser.add_argument(
    "-e", "--encoding_info",
    type=str,
    default="encoding.txt",
    help="The file with label mapping to class name in testing mode or the \
      directory containing encoding files for inference mode ",
    required=True,
  )
  parser.add_argument(
    "-m", "--model_location",
    type=str,
    help="The path to the model file for testing, and the folder with multiple \
      model files for inference",
    required=True,
  )
  parser.add_argument(
    "-b", "--batch_size",
    type=int,
    default=128,
    help="Batch size (default=128).",
  )
  parser.add_argument(
    "-l", "--loss_type",
    type = str,
    default = 'cross_entropy', ## other option is 'focal'
    help='Type of loss function to use for testing (default=cross_entropy)',
    required= False
  )
  parser.add_argument(
    "-d", "--test_mode",
    type = str,
    default = 'testing', ## other options is 'inference'
    help = "argument to specify whether model should be run in testing mode \
      (with imgaes in different class folders) or in inference mode where one\
      image or a folder with images is passed (default:testing)",
    required=False,
  )
  parser.add_argument(
    "-D", "--dir_depth",
    type = int,
    default = 0,
    help = "if a folder F is passed in inference mode, this argument specifies \
      if the images are subfolders of F (1) or in F directly (default:0)",
    required= False,
  )
  parser.add_argument(
    "-o", "--output_dir",
    type = str,
    help = "directory to save log files",
    required=True,
  )
  parser.add_argument(
    "-p", "--prob_threshold",
    type = float,
    default = 0,
    help = "threshold probability for inference mode, if set to 0, no \
      threshold is applied; if 2, use 1/num_classes as threshold; or any value \
      between 0 and 1 (default: 0)",
    required=False,
  )
  parser.add_argument(
    "-L", "--taxon_label_file",
    type = str,
    help = "file with taxon level and associated label in order expected in \
      the model output",
    required=True,
  )
  return parser.parse_args()

def get_data_loaders(data_dir, batch_size):
  '''helper function to apply any transformations/cropping
  and convert batch images into a data loader, used for testing mode'''
  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
  #input_data = datasets.ImageFolder(data_dir, transform=transform)
  # Use this custom class to get the image file paths
  input_data = ImageFolderWithPaths(data_dir, transform=transform)
  input_data_len = len(input_data)
  input_loader = DataLoader(input_data, batch_size=batch_size, shuffle=True, 
                            num_workers=4)
  
  return (input_loader, input_data_len)

def run_testing(label_map, model_ft, image_path, log_file):
  '''function to run model on test data and print out accuracy
  '''
  print("Strat testing")

  print("  load data")
  (test_loader, _) = get_data_loaders(image_path, args.batch_size)
  
  print(f"  # batches:{len(test_loader)}")
  if args.loss_type == 'cross_entropy':
    criterion = LabelSmoothingCrossEntropy()
  # criterion = torch.nn.CrossEntropyLoss()
  elif args.loss_type == 'focal':
    criterion = torch.hub.load(
    'adeelh/pytorch-multi-class-focal-loss',
    model='focal_loss',
    alpha=[1]*args.number_class, ##setting all class weights to 1, can experiment here
    gamma=2, ### hyperparameter to tune
    reduction='mean',
    device='cuda',
    dtype=torch.float32,
    force_reload=False)

  criterion = criterion.to(DEVICE)

  test_loss = 0.0
  class_correct = list(0. for i in range(len(classes)))
  class_total = list(0. for i in range(len(classes)))

  model_ft.eval()

  # feature (data) and label (target) extraction

  for data, target, paths in test_loader:
    #print("target:",target.shape,target)
    if torch.cuda.is_available(): 
      data, target = data.cuda(), target.cuda()
      
    with torch.no_grad():
      output = model_ft(data)
      loss = criterion(output, target)

    test_loss += loss.item()*data.size(0)

    # prediction probability
    #probs          = torch.nn.functional.softmax(output, dim=0)
    # Return max value in the input tensor, 1:dimension to reduce
    _, pred        = torch.max(output, 1)  
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct        = np.squeeze(correct_tensor.numpy()) \
                          if not torch.cuda.is_available() \
                          else np.squeeze(correct_tensor.cpu().numpy())
    # SHS: This else statement will make the last batch not included in the
    #      class total and accuracy calculation, does not make sense.
    '''
    if len(target) == args.batch_size:
      for i in range(args.batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

    else:
      print(f"ERR: # target:{len(target)} != batch_size:{args.batch_size}")
    '''
    for i in range(len(target)):
      label = target.data[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1

  test_loss = test_loss/len(test_loader.dataset)
  print('Test Loss: {:.6f}'.format(test_loss))

  with open(log_file, "w") as f:
    f.write('class,accuracy(%),num_tp,num_total\n')
    for i in range(len(classes)):
      if class_total[i] > 0:
        f.write(str(label_map[classes[i]]) + ',' + \
                '{0:.2f}'.format(100*class_correct[i]/class_total[i]) + ',' + \
                str(np.sum(class_correct[i])) + ',' + \
                str(np.sum(class_total[i])) + '\n')
      else:
        print('Test Accuracy of %5s: N/A (no training examples)' % \
                                                        (label_map[classes[i]]))

    print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
      100. * np.sum(class_correct) / np.sum(class_total),
      np.sum(class_correct), np.sum(class_total)))

  print("Results written to {}".format(log_file))


def run_inference(label_map, model_ft, image_path, log_file, taxon_level, 
                  classes, p_threshold):
  '''function to infer taxa on new images'''
  
  # for output later
  

  print("Start inference")
  if os.path.isfile(image_path):
    pred, ci, _ = identify_taxa(label_map, model_ft, image_path)
    print(f"file: {image_path}, family: {pred}, confidence score: {ci}")

  elif os.path.isdir(image_path):
    # Set probability threshold
    if p_threshold == 2:
      p_threshold = 1/len(taxon_label_dict[taxon_level])
    
    t1 = time()
    with open(log_file, "w") as f:
      # header
      class_str = ','.join(classes)
      f.write(f"file,pred,conf,{class_str}\n")

      # Go through each image
      image_files = os.listdir(image_path)
      for image_file in image_files:
        ext = os.path.splitext(image_file)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
          pred, ci, probs = identify_taxa(label_map, model_ft, 
                                          os.path.join(image_path, image_file))
          ####
          ## IMPORTANT: prob < p_threshold is set to 0
          ####
          prob_list = probs.tolist()
          for idx, prob in enumerate(prob_list):
            if prob < p_threshold:
              prob_list[idx] = 0

          probs_str = ','.join([str(i) for i in prob_list])
          f.write(f"{image_file},{pred},{ci},{probs_str}\n")
        else:
          #print(f"  file:{afile}\tNot an image file")
          pass

      n_img = len(image_files)
      t2    = time()
      print(f"  {n_img} images,", "{0:.2f} sec/image".format((t2-t1)/n_img))
      #print("  results written to {}".format(log_file))
  else:
    print("Path does not exist")

def identify_taxa(label_map, model_ft,image_path):
  '''Advika's identify_family(), modified to return taxa and probs
  '''
  model_ft.eval()
  # Already a global variable
  img = Image.open(image_path)
  transform_norm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])
  # get normalized image
  img_normalized = transform_norm(img).float()
  img_normalized = img_normalized.unsqueeze_(0)
  # input = Variable(image_tensor)
  img_normalized = img_normalized.to(DEVICE)
  # print(img_normalized.shape)
  with torch.no_grad():
    output = model_ft(img_normalized)
    # print(output)
    #score, pred = torch.max(output, 1) 
    probs = torch.nn.functional.softmax(output[0], dim=0)
    # Check the top 1 categories that are predicted.
    top1_prob, top1_catid = torch.topk(probs, 1)

    taxa = label_map[str(classes[top1_catid.item()])]
    conf = top1_prob.item()
    return taxa, conf, probs

def load_encoding(encoding_info):
  '''function to load encoding file'''
  print("Load encodings:")
  print("  may be more than trained, because some classes have only 1 image)")
  encoding_df = pd.read_csv(encoding_info, delimiter = " ", 
                          header=None, names=["label_num", "taxa_name"])
  label_map = dict(zip(encoding_df["label_num"].astype(str), 
                        encoding_df["taxa_name"]))
  print("  # classes in encoding:",len(label_map))

  return label_map

def load_model(model_location):
  print("Load model")
  model_file = Path(model_location)
  model_name = str(model_file).split('/')[-1]
  print(" ", model_name)
  # try:
  model_ft   = torch.jit.load(model_file)
  model_ft   = model_ft.to(DEVICE)
  # except:
  #   print('error finding or loading model')
  #   sys.exit(1)

  return model_ft, model_name

if __name__== '__main__':
  args = parse_arguments()

  DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Using device", DEVICE)
  torch.backends.cudnn.benchmark = True

  # testing images or images for inference
  image_path = args.image_path

  taxon_label_file = Path(args.taxon_label_file)
  taxon_label_dict = {}
  with open(taxon_label_file, "r") as f:
    lines = f.readlines()
    for line in lines:
      [taxon_level, labels] = line.strip().split('\t')
      taxon_label_dict[taxon_level] = labels.split(',')

  # create log directory
  output_dir    = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  if args.test_mode == 'testing':

    model_ft, model_name = load_model(args.model_location)
    label_map = load_encoding(args.encoding_info)
    classes = taxon_label_dict[args.taxon_level]
    
    print("  # classes trained:", len(classes))

    log_file = output_dir / f"log_testing_{model_name}"
    run_testing(label_map, model_ft, image_path, log_file)

  elif args.test_mode == 'inference':

    # probability threshold: 0, 1/num_classes, or any value between 0 and 1
    p_threshold = args.prob_threshold  

    # how deep to traverse the dir: 0 - image in current dir or 1 level deeper
    dir_depth   = args.dir_depth

    # Go through multiple models in a folder
    for model_file in Path(args.model_location).iterdir():
      print("######")
      model_ft, model_name = load_model(model_file)

      # infer the encoding file name
      taxon_level   = model_name.split('_')[0]
      print ("  taxon:", taxon_level)

      encoding_info = Path(args.encoding_info) / f"plants_{taxon_level}-encoding.txt"
      label_map     = load_encoding(encoding_info)
      classes       = taxon_label_dict[taxon_level]
      print("  # classes trained:", len(classes))
      
      # deal with depth information
      if dir_depth == 0:
        log_file = output_dir / f"log_inference_{model_name}"
        run_inference(label_map, model_ft, image_path, log_file, taxon_level, 
                        classes, p_threshold)
      elif dir_depth == 1:
        for subdir in Path(image_path).iterdir():
          subdir_name = str(subdir).split('/')[-1]
          log_file = output_dir / f"log_inference_{model_name}_{subdir_name}"
          run_inference(label_map, model_ft, subdir, log_file, taxon_level, 
                        classes, p_threshold)
      else:
        print(f'ERR: unknown directory depth: {dir_depth}')
        sys.exit(0)








    








  

