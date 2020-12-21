# !/usr/bin/env python

import sys
import os
import gc

gc.enable()
import time

import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
from sklearn.preprocessing import \
    LabelEncoder  # documentation -  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

import torch
from torchvision import transforms  # documentation - https://pytorch.org/docs/stable/torchvision/transforms.html
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset  # documentation - https://pytorch.org/docs/stable/data.html
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torch_optimizer as optim  # documentation - https://pytorch-optimizer.readthedocs.io/en/latest/

import warnings

warnings.filterwarnings("ignore")

# Train Configuration
MIN_SAMPLES_PER_CLASS = 5  # threshold for total number of images in a class. if a class has less than this then it will be discarded from the training set.
BATCH_SIZE = 64
LOG_FREQ = 10
NUM_TOP_PREDICTS = 20
MODEL_PATH = "model2.pt"

# Read Train and Test as pandas data frame
train = pd.read_csv('train_set_kaggle_2020/train/train.csv')
test = pd.read_csv('test_set_kaggle_2019/recognition_solution_v2.1.csv')
# path to train and test directory
train_dir = 'train_set_kaggle_2020/train/'
test_dir = 'test_set_kaggle_2019/'


# Data Loader
class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir: str, mode: str):
        self.df = dataframe
        self.mode = mode
        self.image_dir = image_dir

        transforms_list = []
        if self.mode == 'train':
            # MAYBE CHANGE THIS: increase image size from (64,64) to higher resolutionn. Make sure to change in RandomResizedCrop as well!
            transforms_list = [
                transforms.Resize((64, 64)),  # Resize the input image to the given size.
                transforms.RandomHorizontalFlip(),
                # Horizontally flip the given image randomly with a given probability (deafult: p=0.5).
                transforms.RandomChoice([
                    # Apply single transformation randomly picked from a list, i.e. only one of the following transformation will be applied on a given image.
                    # Crop of random size (default: of 0.08 to 1.0) of the original size and a random aspect ratio
                    # (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to given size (need to be the same size as Resize).
                    transforms.RandomResizedCrop(64),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    # Randomly change the brightness, contrast and saturation of an image.
                    # Random affine transformation of the image keeping center invariant.
                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                            scale=(0.8, 1.2), shear=15,
                                            resample=Image.BILINEAR)
                ]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),  # !!! check and validate those numbers !!!
            ]
        else:  # test mode
            transforms_list.extend([  # ??? why extend and not simply like the train option? ???
                # The resize need to be same as train
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        self.transforms = transforms.Compose(
            transforms_list)  # Composes all the transforms in transforms_list together.

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int):
        image_id = self.df.iloc[index].id
        image_path = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        image = Image.open(image_path)
        image = self.transforms(image)  # apply the chosen transformation on a given image

        if self.mode == 'test':
            return {'image': image,
                    'target': self.df.iloc[index].landmarks}
        else:  # train mode
            return {'image': image,
                    'target': self.df.iloc[index].landmark_id}


# Load Data
def load_data(train, test, train_dir, test_dir):
    counts = train.landmark_id.value_counts()
    selected_classes = counts[
        counts >= MIN_SAMPLES_PER_CLASS].index  # select only classes with minimum amount of objects
    num_classes = selected_classes.shape[0]
    print('classes with at least N samples:', num_classes)

    train = train.loc[train.landmark_id.isin(selected_classes)]
    print('train_df', train.shape)
    print('test_df', test.shape)

    label_encoder = LabelEncoder()  # Encode target labels with value between 0 and N-1. This transformer should be used to encode target values, i.e. y, and not the input x.
    label_encoder.fit(train.landmark_id.values)
    print('found classes', len(label_encoder.classes_))
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


# Optimizer - RAdam https://pytorch-optimizer.readthedocs.io/en/latest/api.html#radam
def radam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.RAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


# Metrics
class AverageMeter:
    # Computes and stores the average and current value
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    # Simplified GAP@1 metric: only one prediction per sample is supported
    # FIXME: ----->>>> MAYBE NEED TO CHANGE THE GAP METRIC METHOD
    # https://www.kaggle.com/davidthaler/gap-metric
    # FIXME: or change to something completely else.
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0]  # FIXME: incorrect, not all test images depict landmarks ???
    return res


# Model - https://pypi.org/project/efficientnet-pytorch/
class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, num_classes):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.avg_pool = nn.AdaptiveAvgPool2d(
            1)  # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        self.output_filter = self.base._fc.in_features
        self.classifier = nn.Linear(self.output_filter,
                                    num_classes)  # Applies a linear transformation to the incoming data

    def forward(self, x):
        x = self.base.extract_features(x)  # extract features based on EfficientNet
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x


# Training
def train_step(train_loader, model, criterion, optimizer, epoch):
    print(f'epoch {epoch}')
    batch_time = AverageMeter()  # AverageMeter is the metrics class
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    num_steps = len(train_loader)

    print(f'total batches: {num_steps}')

    end = time.time()

    original_stdout = sys.stdout
    with open('baseline epoch=5 min_sample=5.txt', 'w') as f:
        sys.stdout = f
        print("\nepoch    batch             time    (avg time)        loss   (avg loss)    GAP score   (avg score)")
        print(
            "---------------------------------------------------------------------------------------------------------")
        for i, data in enumerate(train_loader):
            input_ = data['image']
            target = data['target']
            batch_size, _, _, _ = input_.shape

            output = model(input_.cuda())
            loss = criterion(output, target.cuda())
            confs, predicts = torch.max(output.detach(), dim=1)
            avg_score.update(GAP(predicts, confs, target))
            losses.update(loss.data.item(), input_.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % LOG_FREQ == 0:
                print(f'{epoch}     |  [{i}/{num_steps}]\t||'
                      f' {batch_time.val:.3f} | ({batch_time.avg:.3f})\t||'
                      f' {losses.val:.4f} | ({losses.avg:.4f})\t||'
                      f' {avg_score.val:.4f}   |   ({avg_score.avg:.4f})\t||')

        print(f' * average GAP on train {avg_score.avg:.4f}')
        sys.stdout = original_stdout

    torch.save(model.state_dict(), MODEL_PATH)


# Inference (evaluation) Function
def inference(data_loader, model, label_encoder):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            input_, target = data['image'], data['target']
            output = model(input_.cuda())
            output = activation(output)

            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)
            all_confs.append(confs)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    predicts_, confs_ = predicts.cpu().numpy(), confs.cpu().numpy()
    labels = [label_encoder.inverse_transform(pred) for pred in predicts_]
    print('labels')
    print(np.array(labels))
    print('confs')
    print(np.array(confs_))


# Main
if __name__ == '__main__':
    global_start_time = time.time()
    train_loader, test_loader, label_encoder, num_classes = load_data(train, test, train_dir, test_dir)

    model = EfficientNetEncoderHead(depth=0, num_classes=num_classes)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = radam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=1e-4)

    for epoch in range(1, 6):
        print('!' * 75)
        train_step(train_loader, model, criterion, optimizer, epoch)

    # print('inference mode')
    # inference(test_loader, model, label_encoder)