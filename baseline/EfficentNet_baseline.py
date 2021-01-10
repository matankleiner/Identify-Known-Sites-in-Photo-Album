# !/usr/bin/env python3.6

import sys
import os
import gc
gc.enable()

import time
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
from sklearn.preprocessing import LabelEncoder  # documentation -  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
import torch
from torchvision import transforms  # documentation - https://pytorch.org/docs/stable/torchvision/transforms.html
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset  # documentation - https://pytorch.org/docs/stable/data.html
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torch_optimizer as optim  # documentation - https://pytorch-optimizer.readthedocs.io/en/latest/

import warnings
warnings.filterwarnings("ignore")

# based on the code - https://www.kaggle.com/rhtsingh/pytorch-training-inference-efficientnet-baseline by sourcecode369

# Train Configuration 
MIN_SAMPLES_PER_CLASS = 10  # threshold for total number of images in a class. if a class has less than this then it will be discarded from the training set.
BATCH_SIZE = 64
LOG_FREQ = 10
EPOCHS = 10
MODEL_PATH = "min_samples_per_class_" + str(MIN_SAMPLES_PER_CLASS) + "_" + str(EPOCHS) + "epochs.pt"
LOSS_PATH = "loss_min_samples_per_class_" + str(MIN_SAMPLES_PER_CLASS) + "_" + str(EPOCHS) + "epochs.csv"
TRAINING_PATH = "baseline training process with min_samples_per_class_" + str(MIN_SAMPLES_PER_CLASS) + "_" + str(EPOCHS) + "epochs.txt" 
RESULT_PATH = "results_min_samples_per_class_" + str(MIN_SAMPLES_PER_CLASS) + "_" + str(EPOCHS) + "epochs.txt" 

# Read Train and Test as pandas data frame
train = pd.read_csv('train_set_kaggle_2020/train/train.csv')
test = pd.read_csv('test_set_kaggle_2019/recognition_solution_v2.1.csv')
# path to train and test directory
train_dir = 'train_set_kaggle_2020/train/'
test_dir = 'test_set_kaggle_2019/'

class ImageDataset(Dataset):
    """ Image dataset class """
    def __init__(self, dataframe, image_dir, mode):
        """
        Initialization of the class. Chosing different transformation to apply on the data set images if mode is 'train' or 'test'
        Param:
            dataframe (pd.DataFrame): Dataframe made of the train/test set csv file
            image_dir (string): Path to the image directory
            mode (string): Set the mode of the code, can be 'train' or 'test'
        """
        self.df = dataframe
        self.mode = mode
        self.image_dir = image_dir

        transforms_list = []
        if self.mode == 'train':
            transforms_list = [
                transforms.Resize((64, 64)),  # Resize the input image to the given size.
                transforms.RandomHorizontalFlip(), # Horizontally flip the given image randomly with a given probability (deafult: p=0.5).
                transforms.RandomChoice([ # Apply only one from the following (randomly picked)
                    # Crop of random size of the original size and a random aspect ratio
                    # of the original aspect ratio is made. This crop is finally resized to given size
                    # (need to be the same size as Resize).
                    transforms.RandomResizedCrop(64),
                    # Randomly change the brightness, contrast and saturation of an image.
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    # Random affine transformation of the image keeping center invariant.
                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
                ]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),  
            ]
        else:  # test mode
            transforms_list.extend([  
                transforms.Resize((64, 64)), # The resize need to be same as train
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        self.transforms = transforms.Compose(transforms_list)  # Composes all the transforms in transforms_list together.

    def __len__(self):
        """
        Return:
            (int): The length of the dataset
        """
        return self.df.shape[0]

    def __getitem__(self, index):
        """
        Get an image from the data set
        Param:
            index (int): Index of the image we want to get
        Return:
            (Dictionary): Each image path. If 'train' mode also each image class.
        """
        image_id = self.df.iloc[index].id
        image_path = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg" # image path as image divided into different directories
        image = Image.open(image_path)
        image = self.transforms(image)  # apply the chosen transformation on a given image

        if self.mode == 'test':
            return {'image': image}
        else:  # train mode
            return {'image': image,
                    'target': self.df.iloc[index].landmark_id}

def load_data(train, test, train_dir, test_dir):
    """
    Selecting only classes with MIN_SAMPLES_PER_CLASS. Encode the labels using LabelEncoder.
    Load datat using DataLoader
    Param:
        train (pd.DataFrame): Dataframe made of the train set csv file
        test (pd.DataFrame): Dataframe made of the test set csv file
        train_dir (string): path to train directory
        test_dir (string): path to test directory
    Return:
          train_loader (DataLoader): The train set data loader
          test_loader (DataLoader): The test set data loader
          label_encoder (LabelEncoder): The labels normalized
          num_classes (int): Number of classes with at least MIN_SAMPLES_PER_CLASS
    """
    counts = train.landmark_id.value_counts() # value_counts() return a Series containing counts of unique values
    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index  # select only classes with minimum amount of objects
    num_classes = selected_classes.shape[0]
    print(f'Classes with at least {MIN_SAMPLES_PER_CLASS} samples:', num_classes)

    train = train.loc[train.landmark_id.isin(selected_classes)]
    print(f'train_df (only classes with {MIN_SAMPLES_PER_CLASS} samples) size: ', train.shape)
    print('test_df size: ', test.shape)

    # Encode target labels with value between 0 and N-1. This transformer should be used to encode target values, i.e. y, and not the input x.
    label_encoder = LabelEncoder()
    label_encoder.fit(train.landmark_id.values)
    print('Found classes: ', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == num_classes

    train.landmark_id = label_encoder.transform(train.landmark_id)  # Transform labels to normalized encoding.
    
    train_dataset = ImageDataset(train, train_dir, mode='train')
    test_dataset = ImageDataset(test, test_dir, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)

    return train_loader, test_loader, label_encoder, num_classes

def radam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    """
    The chosen optimizer - RAdam
    Info: https://pytorch-optimizer.readthedocs.io/en/latest/api.html#radam
    Paper: https://arxiv.org/abs/1908.03265
    Param:
        parameters: The chosen model paramters
        lr (int), betas (tuple), eps (int), weight_decay (int): learning paramters
    Return:
          RAdam optimizer instance with the given parameters
    """
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.RAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

class AverageMeter:
    """ Average Meter - Computes and stores the average and current value """

    def __init__(self):
        """
        Initialization of the class with reset.
        """
        self.reset()

    def reset(self):
        """
        Reset all values to 0
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n = 1):
        """
        Update all values.
        Param:
            val (float): Value to update
            n (int): Size of the values needed update
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EfficientNetEncoderHead(nn.Module):
    """ 
    EfficentNet Encoder Head
    Info: https://pypi.org/project/efficientnet-pytorch/
    Paper: https://arxiv.org/pdf/1905.11946.pdf
    """
    def __init__(self, depth, num_classes):
        """
        Initialization of the class.
        Param:
            depth (int): Chosing the efficentnet base one want to use, values range from 0 to 7 (include). The bigget
                         the depth the higher the paramters number and the highet the accuracy achieved. All models are pretrained
            num_classes (int): The number of classes in the train set after filtering classes with less than MIN_SAMPLES_PER_CLASS samples
        """
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        self.output_filter = self.base._fc.in_features
        self.classifier = nn.Linear(self.output_filter, num_classes) # Applies a linear transformation to the incoming data

    def forward(self, x):
        """
        Forward Step
        Param:
            x
        Return:
            x
        """
        x = self.base.extract_features(x) # extract features based on EfficientNet
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x


def train_step(train_loader, model, criterion, optimizer):
    """
    Training step.
    Param:
        train_loader (DataLoader): The train set Data loader
        model (model): EfficientNet instance
        criterion (torch.nn): Instance of the chosen loss criterion (we chose CrossEntropyLoss)
        optimizer (optimizer): Instance of the chosen optimizer (we chose RAdam)
    """
    original_stdout = sys.stdout
    with open(TRAINING_PATH, 'w') as f: # write learning process to file
      sys.stdout = f
      batch_time = AverageMeter() # AverageMeter is for updating the learning process
      losses = AverageMeter()
      loss_process = []
      model.train()
      num_steps = len(train_loader)
      print(f'total batches: {num_steps}')

      begin = time.time() #time update
      for epoch in range(1,EPOCHS+1):
          print(f"Epoch: {epoch}")
          print("-" * 70)
          for i, data in enumerate(train_loader):
              input_ = data['image']
              target = data['target']
              batch_size, _, _, _ = input_.shape
              output = model(input_.cuda())
              loss = criterion(output, target.cuda())
              confs, predicts = torch.max(output.detach(), dim=1)
              losses.update(loss.data.item(), input_.size(0))
              # learn
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
      
              batch_time.update(time.time() - begin)
              begin = time.time()
      
              if i % LOG_FREQ == 0:
                  print(f'batch: [{i}/{num_steps}] batch time: {batch_time.val:.3f} trainign loss: {losses.val:.4f} avg training loss: {losses.avg:.4f}')
                  loss_process.append(losses.avg)
                  
      sys.stdout = original_stdout
    
    # convert loss process into dataframe and export it as csv file 
    loss_df = pd.DataFrame(data = loss_process)  
    loss_df.to_csv(LOSS_PATH)
    torch.save(model.state_dict(), MODEL_PATH) # save the learned model
    print("Saved model @", MODEL_PATH)
    
def inference(data_loader, model, label_encoder):
    """
    Evaluation.
    Param:
        data_loader (DataLoader): The test set Data loader
        model (model): EfficientNet instance
        label_encoder (LabelEncoder): label_encoder as returned from @func load_data
    """
    model.load_state_dict(torch.load(MODEL_PATH)) # load the learned model
    print("Loaded model @", MODEL_PATH)
    model.eval()

    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []
    results = {}
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            input_ = data['image']
            output = model(input_.cuda())
            output = activation(output)           
            confs, predicts = torch.topk(output, 5) # top 5 confidence and predicts of the network
            predicts_, confs_ = predicts.cpu().numpy(), confs.cpu().numpy()
            labels = [label_encoder.inverse_transform(pred) for pred in predicts_] 
            results[i] = labels[0]

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(RESULT_PATH) # save results as CSV file

if __name__ == '__main__':
    #global_start_time = time.time()
    train_loader, test_loader, label_encoder, num_classes = load_data(train, test, train_dir, test_dir)
    model = EfficientNetEncoderHead(depth=0, num_classes=num_classes)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = radam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=1e-4)

    print("Training Mode ---> Write training process to file")
    train_step(train_loader, model, criterion, optimizer)
        
    #print('Evaluation Mode')
    #inference(test_loader, model, label_encoder)
    

    