import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units: list, hidden_layer_activation='relu', init_type='uniform'):
        super(MLP, self).__init__()
        self.units = units
        self.n_layers = len(units) # including input and output layers
        valid_activations = {'relu': nn.ReLU(),
                             'tanh': nn.Tanh(),
                             'sigmoid': nn.Sigmoid()}
        self.activation = valid_activations[hidden_layer_activation]
        #####################################################################################
        # TODO: Implement the model architecture with respect to the units: list            #
        # use nn.Sequential() to stack layers in a for loop                                 #
        # It can be summarized as: ***[LINEAR -> ACTIVATION]*(L-1) -> LINEAR -> SOFTMAX***  #
        # Use nn.Linear() as fully connected layers                                         #
        #####################################################################################
        self.mlp = nn.Sequential()


        #####################################################################################
        #                                 END OF YOUR CODE                                  #
        #####################################################################################

    def forward(self, X):
        #####################################################################################
        # TODO: Forward propagate the input                                                 #
        # ~ 2 lines of code#
        # First propagate the input and then apply a softmax layer                          #
        #####################################################################################

        #####################################################################################
        #                                 END OF YOUR CODE                                  #
        #####################################################################################
        return out

        

