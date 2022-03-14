from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_transformations(img_size):
    #####################################################################################
    # TODO: Implement appropriate transformation which needed to be applied on images   #
    # Use torchvision.transforms.transforms.Compose to stack several transformations    #
    # Visit https://pytorch.org/vision/stable/transforms.html for more                  #
    #####################################################################################
    image_transforms = None


    #####################################################################################
    #                                 END OF YOUR CODE                                  #
    #####################################################################################
    return image_transforms


def get_one_hot(label, num_classes):
    #####################################################################################
    # TODO: Implement a function to compute the one hot encoding for a single label     #
    # label --  (int) Categorical labels                                                #
    # num_classes --  (int) Number of different classes that label can take             #
    # Hint: Use torch.nn.functional                                                     #
    #####################################################################################
    one_hot_encoded_label = None


    #####################################################################################
    #                                 END OF YOUR CODE                                  #
    #####################################################################################
    return one_hot_encoded_label


def visualize_samples(dataset, n_samples, cols=4):
    #####################################################################################
    # TODO: Implement a function to visualize first n_samples of your dataset           #
    # dataset --  (int) Categorical labels                                              #
    # Your plot must be a grid of images with title of their labels                     #
    # Note: use torch.argmax to decode the one-hot encoded labels back to integer labels#
    # Note: You may need to permute the image of shape (C, H, W) to (H, W, C)           #
    #####################################################################################
    rows = n_samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))

    #####################################################################################
    #                                 END OF YOUR CODE                                  #
    #####################################################################################


def init_weights(net: nn.Module, init_type='zero_constant'):
    #####################################################################################
    # TODO: A function that initializes the weights in the entire nn.Module recursively.#
    # When you get an instance of your nn.Module model later, pass this function        #
    # to torch.nn.Module.apply. For more explanation visit:                             #
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch #
    # Note: initialize both weights and biases of the entire model                      #
    #####################################################################################
    valid_initializations = ['zero_constant', 'uniform']
    if init_type not in valid_initializations:
        print("INVALID TYPE OF WEIGHT INITIALIZATION!")
        return

    #####################################################################################
    #                                 END OF YOUR CODE                                  #
    #####################################################################################