# BloodMNIST_CNN
A CNN classifier architecture on the BloodMNIST part of the MedMNIST dataset.

This document looks at implementing classifier models via supervised learning to correctly classify images. I have used images from the MedMNIST dataset which contains a range of health related image datasets that have been designed to match the shape of the original digits MNIST dataset. Specifically I worked with the BloodMNIST part of the dataset. The data file will be loaded as a dictionary that contains both the images and labels already split to into training, validation and test sets. The each sample is a 28 by 28 RGB image and are not normalised.  I trained a CNN classifier architecture on this dataset and compare their performance.
