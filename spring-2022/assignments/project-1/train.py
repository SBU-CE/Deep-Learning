import os
import argparse
import torch
import torch.nn as nn
from dataset import SignDigitDataset
from torch.utils.data import DataLoader
from utils import *
from model import MLP
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sign_digits_experiment_1')

parser = argparse.ArgumentParser()
# Hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100, required=True, help='number of epochs for training')
parser.add_argument('--print_every', type=int, default=10, help='print the loss every n epochs')
parser.add_argument('--img_size', type=int, default=64, help='image input size')
parser.add_argument('--n_classes', type=int, default=6, help='number of classes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_layers', type=int, required=True, nargs='+',
                    help='number of units per layer (except input and output layer)')
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh'], help='activation layers')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

#####################################################################################
# TODO: Complete the script to do the following steps                               #
# 0. Create train/test datasets
# 1. Create train and test data loaders with respect to some hyper-parameters       #
# 2. Get an instance of your MLP model.                                             #
# 3. Define an appropriate loss function (e.g. cross entropy loss)                  #
# 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
# 5. Implement the main loop function with n_epochs iterations which the learning   #
#    and evaluation process occurred there.                                         #
# 6. Save the model weights                                                         #
#####################################################################################


# 0. creating train_dataset and test_dataset

# 1. Data loaders

# 2. get an instance of the model

# 3, 4. loss function and optimizer


# 5. Train the model
n_train_batches = 0
n_test_batches = 0
for epoch in range(args.n_epochs):
    train_running_loss, test_running_loss = 0.0, 0.0


    # ...log the running loss
    writer.add_scalar('Train Loss', train_running_loss / n_train_batches, epoch)
    writer.add_scalar('Test Loss', test_running_loss / n_test_batches, epoch)

    if epoch % args.print_every == 0:
        # You have to log the accuracy as well
        print('Epoch [{}/{}]:\t Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1,
                                                                               args.n_epochs,
                                                                               train_running_loss / n_train_batches,
                                                                               test_running_loss / n_test_batches))

#####################################################################################
#                                 END OF YOUR CODE                                  #
#####################################################################################


# save the model weights
checkpoint_dir = 'checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)