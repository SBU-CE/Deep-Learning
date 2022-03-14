import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from utils import *


class SignDigitDataset(Dataset):
    def __init__(self, root_dir='data/', h5_name='train_signs.h5',
                 train=True, transform=None, initialization=None):
        self.transform = transform
        self.train = train

        # h5 file related attributes
        self.h5_path = os.path.join(root_dir, h5_name)
        key = 'train_set' if self.train else 'test_set'
        self.dataset_images = np.array(self._read_data(self.h5_path)[key+'_x'])
        self.dataset_labels = np.array(self._read_data(self.h5_path)[key + '_y'])


    def __len__(self):
        return len(self.dataset_images)

    def __getitem__(self, index):
        #####################################################################################
        # TODO: Use the image and label transformations you implemented to prepare a sample #
        # First transform the image                                                         #
        # And then use utils.get_one_hot() to create the proper label for model             #
        #####################################################################################
        img = None
        label = None

        #####################################################################################
        #                                 END OF YOUR CODE                                  #
        #####################################################################################
        return {
            'image': img,
            'label': label
        }

    def _read_data(self, h5_path):
        dataset = h5py.File(h5_path, "r")

        return dataset