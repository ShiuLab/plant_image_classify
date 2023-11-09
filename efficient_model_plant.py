import argparse
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

#!pip install timm
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import create_transform

import warnings
warnings.filterwarnings("ignore")
import os
import sys
from tqdm import tqdm
import time
import copy
import matplotlib.pyplot as plt
#import seaborn as sns

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--data_location",
    type=str,
    default=os.path.expanduser('/mnt/research/xprize23/plants_test/family/augmented_family_train'),
    help="The root directory for the training datasets.",
  )
  parser.add_argument(
    "--validation_data_location",
    type=str,
    default=os.path.expanduser('/mnt/research/xprize23/plants_test/family/augmented_family_validation'),
    help="The root directory for the validation datasets.",
  )
  parser.add_argument(
    "--test_data_location",
    type=str,
    default=os.path.expanduser('/mnt/research/xprize23/plants_test/family/family_test'),
    help="The root directory for the test datasets.",
  )
  parser.add_argument(
    "--timm_model",
    type=str,
    default="tf_efficientnetv2_b0",
    help="timm model to finetune (default: tf_efficientnetv2_b0)",
  )
  parser.add_argument(
    "--model_location",
    type=str,
    default=os.path.expanduser('/mnt/research/xprize23/plants_test/family/model_selection/'),
    help="Where to save the models.",
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
  )
  parser.add_argument(
    "--custom-template", action="store_true", default=False,
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=4,
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=10,
  )
  # parser.add_argument(
  #   "--warmup-length",
  #   type=int,
  #   default=500,
  # )
  # parser.add_argument(
  #   "--continue_from_saved",
  #   action='store_true',
  #   default=False,
  # )
  # parser.add_argument(
  #   "--continue_from",
  #   type=int,
  #   default=30,
  # )
  parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help ="learning rate"
  )
  # parser.add_argument(
  #   "--wd",
  #   type=float,
  #   default=0.1,
  # )
  # parser.add_argument(
  #   "--model",
  #   default='ViT-B/32',
  #   help='Model to use -- you can try another like RN50,N101,RN50x4,RN50x16,RN50x64,ViT-B/32,ViT-B/16,ViT-L/14,ViT-L/14@336px'
  # )
  parser.add_argument(
    "--ckpt_name_prefix",
    default='ckpt',
    help='Prefix for the checkpoint filename'
  )
  # parser.add_argument(
  #   "--timm-aug", action="store_true", default=False,
  # )
  # parser.add_argument(
  #   "--number_class",
  #   type = int,
  #   default = 256,
  #   required= True
  # )
  # parser.add_argument(
  #   "--saved_model_dir",
  #   type = str,
  #   required= False
  # )
  parser.add_argument(
    "--loss_type",
    type = str,
    default = 'cross_entropy', ## other option is 'focal'
    required= False
  )
  return parser.parse_args()

def get_classes(data_dir):
  all_data = datasets.ImageFolder(data_dir)
  return all_data.classes

# def get_data_loaders(data_dir, batch_size, train = False):
#   if train:
#     transform = transforms.Compose([
#       transforms.RandomHorizontalFlip(p=0.5),
#       #transforms.RandomVerticalFlip(p=0.5),
#       #transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
#       transforms.Resize(256),
#       transforms.CenterCrop(224),
#       transforms.ToTensor(),
#       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#           ])
#     train_data = datasets.ImageFolder(data_dir, transform=transform)
#     train_data_len = int(len(train_data))
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
#     return train_loader, train_data_len
  
#   else:
#     transform = transforms.Compose([
#       transforms.Resize(256),
#       transforms.CenterCrop(224),
#       transforms.ToTensor(),
#       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
#     tv_data = datasets.ImageFolder(data_dir, transform=transform)
#     tv_data_len = int(len(tv_data))
#     tv_loader = DataLoader(tv_data, batch_size=batch_size, shuffle=True, num_workers=4)
#     return tv_loader, tv_data_len

def get_data_loaders(data_dir, batch_size):

    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    input_data = datasets.ImageFolder(data_dir, transform=transform)
    input_data_len = len(input_data)
    input_loader = DataLoader(input_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return (input_loader, input_data_len)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0
      # Iterate over data.
      for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      
      if phase == 'train':
        training_history['accuracy'].append(epoch_acc)
        training_history['loss'].append(epoch_loss)
      elif phase == 'val':
        validation_history['accuracy'].append(epoch_acc)
        validation_history['loss'].append(epoch_loss)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        print("saving best model so far")
        model.load_state_dict(best_model_wts)
        example = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model.cpu(), example)

        ckpt_name = f"{args.ckpt_name_prefix}-{args.timm_model}.pth"
        traced_script_module.save(f"{args.model_location}{ckpt_name}")
        model = model.to(DEVICE)

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model


if __name__== '__main__':
  args = parse_arguments()
  (train_loader, train_data_len) = get_data_loaders(args.data_location, args.batch_size)
  (val_loader, valid_data_len) = get_data_loaders(args.validation_data_location, args.batch_size)
  (test_loader, test_data_len) = get_data_loaders(args.test_data_location, args.batch_size)
  classes = get_classes(args.data_location)
  dataloaders = {
  "train":train_loader,
  "val": val_loader
  }
  dataset_sizes = {
    "train":train_data_len,
    "val": valid_data_len
  }
  print(len(train_loader))
  print(len(val_loader))
  print(len(test_loader))

  DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.backends.cudnn.benchmark = True

  if not os.path.exists(args.model_location):
    os.makedirs(args.model_location)

  model = timm.create_model(args.timm_model, pretrained=True)

  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.classifier.in_features

  model.classifier = nn.Sequential(
    nn.Linear(n_inputs,2048),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(2048, len(classes))
  )

  model = model.to(DEVICE)
  print(model.classifier)


  if args.loss_type == 'cross_entropy':
    criterion = LabelSmoothingCrossEntropy()
    # criterion = torch.nn.CrossEntropyLoss()
  elif args.loss_type == 'focal':
    criterion = torch.hub.load(
    'adeelh/pytorch-multi-class-focal-loss',
    model='focal_loss',
    alpha=[1]*len(classes), ##setting all class weights to 1, can experiment here
    gamma=2, ### hyperparameter to tune
    reduction='mean',
    device='cuda',
    dtype=torch.float32,
    force_reload=False
    )
# criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(DEVICE)
  optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

  training_history = {'accuracy':[],'loss':[]}
  validation_history = {'accuracy':[],'loss':[]}

  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)  

  model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
             num_epochs=args.epochs)

  test_loss = 0.0
  class_correct = list(0. for i in range(len(classes)))
  class_total = list(0. for i in range(len(classes)))

  model_ft.eval()

  for data, target in tqdm(test_loader):
    if torch.cuda.is_available(): 
      data, target = data.cuda(), target.cuda()
    with torch.no_grad():
      output = model_ft(data)
      loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)  
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) \
                if not torch.cuda.is_available() \
                else np.squeeze(correct_tensor.cpu().numpy())
    if len(target) == args.batch_size:
      for i in range(args.batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

  test_loss = test_loss/len(test_loader.dataset)
  print('Test Loss: {:.6f}\n'.format(test_loss))

  for i in range(len(classes)):
    if class_total[i] > 0:
      print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
        classes[i], 100 * class_correct[i] / class_total[i],
        np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
      print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

  print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
  100. * np.sum(class_correct) / np.sum(class_total),
  np.sum(class_correct), np.sum(class_total)))

