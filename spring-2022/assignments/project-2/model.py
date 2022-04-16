import torch.nn as nn


class DownConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels
    ):
        super(DownConv, self).__init__()

        # properties of class

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        None


class UpConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels
    ):
        super(UpConv, self).__init__()

        # properties of class

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        None


class Bottleneck(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels
    ):
        super(Bottleneck, self).__init__()

        # properties of class

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        None


class BaseModel(nn.Module):
    def __init__(
            self, kernel, num_filters, num_colors, in_channels=1, padding=1
    ):
        super(BaseModel, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Other properties if needed

        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        None



class CustomUNET(nn.Module):
    def __init__(
            self, num_filters, num_colors, in_channels=1, out_channels=3
    ):
        super(CustomUNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Other properties if needed

        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        None

