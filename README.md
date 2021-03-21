# Identify Known Sites in Photo Album
 
[![shield](https://img.shields.io/badge/machine-learning-purple)](https://memegenerator.net/instance/55888623/x-x-everywhere-machine-learning-machine-learning-everywhere)
[![shield](https://img.shields.io/badge/python-3.6-blue)](https://www.python.org/)
[![shield](https://img.shields.io/badge/torch-1.8.0-green)](https://pytorch.org/)
[![shield](https://img.shields.io/badge/torchvision-0.9.0-green)](http://pytorch.org/vision/stable/index.html)
[![shield](https://img.shields.io/badge/pandas-0.25.0-green)](https://pandas.pydata.org/)
[![shield](https://img.shields.io/badge/numpy-1.19.5-green)](https://numpy.org/)
[![shield](https://img.shields.io/badge/opencv-0.25.0-green)](https://opencv.org/)
[![shield](https://img.shields.io/badge/matplotlib-1.19.5-green)](https://matplotlib.org/)
[![shield](https://img.shields.io/badge/seaborn-0.11.0-green)](https://seaborn.pydata.org/)
[![shield](https://img.shields.io/badge/efficientnet_pytorch-0.7.0-green)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![shield](https://img.shields.io/badge/torch_optimizer-0.1.0-green)](https://pypi.org/project/torch-optimizer/)
[![shield](https://img.shields.io/badge/sklearn-0.21.3-green)](https://scikit-learn.org/stable/)
[![shield](https://img.shields.io/badge/PIllow-6.1.0-green)](https://pillow.readthedocs.io/en/stable/)
[![shield](https://img.shields.io/badge/tqdm-4.55.0-green)](https://github.com/tqdm/tqdm)
[![shield](https://img.shields.io/badge/yolo-v3-yellow)](https://pjreddie.com/darknet/yolo/)
[![shield](https://img.shields.io/badge/yolo-v4-yellow)](https://github.com/AlexeyAB/darknet)
[![shield](https://img.shields.io/badge/GLD-v2-red)](https://storage.googleapis.com/gld-v2/web/index.html)
[![shield](https://img.shields.io/badge/OpenImagesDataset-v4-red)](https://storage.googleapis.com/openimages/web/factsfigures_v4.html)
[![shield](https://img.shields.io/badge/COCO-Dataset-red)](https://storage.googleapis.com/gld-v2/web/index.html)

![alt text](https://github.com/matankleiner/Identify-Known-Sites-in-Photo-Album/blob/master/images/project_scheme.gif)

## Introduction:

This is a university project based on the [Google Landmark Recognition 2020 kaggle competiton.](https://www.kaggle.com/c/landmark-recognition-2020)

The goal of this project is to classify successfully images of known sites from around the world, given big and challenging train set to learn from and a test set that contain mainly out of domain images.  

In face of the special and challenging features of the data set, we proposed and implemented two possible solution using machine learning techniques.

The first solution, a baseline, is a simple straight forward aprrocah, training a CNN (EfficientNet using RAdam optimizer) and use it as a classifier. This solution faile to overcome the challenging aspects of the data set and yields poor results.

The second solution is a retrival based soultion that derive inspiration from other teams soultion to this competition.

This solution consist of two steps, the first is to clean the test set from out of domain images using object detection (we used [YOLO darknet](https://github.com/AlexeyAB/darknet) implementation). Object detection examples: 

![alt text](https://github.com/matankleiner/Identify-Known-Sites-in-Photo-Album/blob/master/landmark_classifier/example_images/predictions1.jpg)
![alt text](https://github.com/matankleiner/Identify-Known-Sites-in-Photo-Album/blob/master/results_and_evaluation/7f15d65c538fd83b_62916/predictions_v3.jpg)

The second is classification using nearest neighbor algorithm, using the images features vector. The power of using feature vectore and K-NN (the test set image is to the left, next to it there are the 5 nearest neighbors from the train set): 

![alt text](https://github.com/matankleiner/Identify-Known-Sites-in-Photo-Album/blob/master/results_and_evaluation/fde4d840e5f7ae90_23777/23777_nn.png)
![alt text](https://github.com/matankleiner/Identify-Known-Sites-in-Photo-Album/blob/master/results_and_evaluation/7e77ce1f29338f90_18679/18679_nn.png)

This soultuin is built to face on the challenging features of the data set and although the solution it yields are far from great they are much better than the baseline's results.  

## Code 
The code we wrote for this project is organized in sub directories, so that there is a sub directory for each part of the project.
Each sub directory contain the relevant code files (.py or .ipynb) and may contain csv files or images. 

We tried to write the code so it will be organized and well documented. 

## Prerequisites

To run the whole code of this project, one needs the following libraries (in the specified version or higher):
| Library | Version |
| ------------- | ------------- |
| Python | 3.6 |
| torch | 1.8.0 |
| torchvision | 0.9.0 |
| pandas | 1.25.0 |
| numpy | 1.19.0 |
| opencv | 4.2.0 |
| matplotlib | 3.2.1 |
| seaborn | 0.11.0 |
| efficientnet_pytorch | 0.7.0 |
| torch_optimizer | 0.1.0 |
| sklearn | 0.21.3 |
| PIlow | 6.1.0 |
| tqdm | 4.55.0 |

In this project we also used [YOLO darknet](https://github.com/AlexeyAB/darknet) implementation as an object detector. We used version 3 and version 4 network that were pre trained on [Open Images Dataset](https://storage.googleapis.com/openimages/web/factsfigures_v4.html) and [COCO Dataset](https://storage.googleapis.com/gld-v2/web/index.html) accordingly. 

Many of the code in this project is part of a [jupyter notebook](https://jupyter.org/). Unfortunately, GitHub is not able to render successfully all the notebooks, so one can download them and run them locally or via colab. 

## Team:

Matan Kleiner 

Yuval Snir 

under the guidance of Ori Linial

## References

[1] T. Weyand, A. Araujo, B. Cao and J. Sim, Proc. "Google Landmarks Dataset v2 - A Large-Scale Benchmark for 	Instance-Level Recognition and Retrieval", CVPR'20

[2] K. Chen et-al “2nd Place and 2nd Place Solution to Kaggle Landmark 	Recognition and Retrieval Competition 2019", arXiv:1906.03990 	[cs.CV], Jun. 2019.

[3] J. Redmon and A. Farhadi. "YOLOv3: An Incremental Improvement", arXiv:1804.02767v1 [cs.CV] Apr. 2018.

[4] A. Krizhevsky, I. Sutskever, and G. E. Hinton. "ImageNet classification with deep convolutional neural networks", In Proceedings of NIPS, pages 1106–1114, 2012.



