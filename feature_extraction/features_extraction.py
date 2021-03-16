# !/usr/bin/env python3.6

"""
In this code we'll take the test and train set images and turn them into feature vector.
We'll use torchvision's ResNet-18 pre-trained model last pooling layer.
"""

import torch
import torchvision
import torchvision.models as models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Read Train and Test as pandas data frame
train_df = pd.read_csv('train_set_kaggle_2020/train/train.csv')
test_df = pd.read_csv('test_set_kaggle_2019/test.csv')
# path to train and test directory
train_dir = 'train_set_kaggle_2020/train/'
test_dir = 'test_set_kaggle_2019/'

model = models.resnet18(pretrained=True) # Load the pretrained model
layer = model._modules.get('avgpool') # Select the avgpool last layer
model.eval() # Set model to evaluation mode (ensure there are no active dropout layers during forward pass)
VEC_LEN = 512 # The ResNet-18 'avgpool' layer has an output size of 512

# Transformation for ResNet-18
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

def img2vec(image):
    """
    Transform an image to a feature vector
    Param:
        image (2d-list): An image as a 2d-list
    Return:
        embedding (list): an embedded feature vectors of the image
    """
    t_img = transforms(image) # Apply the chosen transformation on the image, return a PyTorch tensor
    embedding = torch.zeros(VEC_LEN) # Create a vector of zeros in the wanted length

    def copy_data(m, i, o):
        """
        A function that will copy the output of a layer.
        Using flatten to correct dimensions.
        """
        embedding.copy_(o.flatten())

    attached = layer.register_forward_hook(copy_data) # Attach that function to our selected layer

    # Run the model on the transformed image, no_grad needed
    with torch.no_grad():
        model(t_img.unsqueeze(0))
    attached.remove() # Detach the copy function from the layer

    return embedding

"""
# Convert all the train set images to feature vectors
embedded_data = []
labels = []
print("Converting the train images to feature vectors:")
for i in tqdm(range(train_df.shape[0])):
    image_id = train_df["id"][i]
    image_path = f"{train_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
    image = Image.open(image_path)
    feature_vector = img2vec(image)
    embedded_data.append(feature_vector.numpy()) # Save the feature vector of each image
    labels.append( train_df["landmark_id"][i]) # Save the label of each image

torch.save(embedded_data111, 'embedded_data.pt')
torch.save(labels111, 'labels.pt')
"""

# Convert all the test set images to feature vectors
embedded_test = []
print("Converting the test images to feature vectors:")
for i in tqdm(range(test_df.shape[0])):
    image_id = test_df["id"][i]
    image_path = f"{test_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
    image = Image.open(image_path)
    feature_vector = img2vec(image)
    embedded_test.append(feature_vector.numpy()) # Save the feature vector of each image

torch.save(embedded_test, 'embedded_test.pt')



