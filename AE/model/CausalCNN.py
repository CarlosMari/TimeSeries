"""
The following causal_cnn file is directly copied from Franchesci et. al. Original source code is available at 
https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries.
"""

import torch
import torch.nn as nn
from AE.model.utils import Chomp, SqueezeChannels, get_activation_func


class CausalConvolutionBlock(nn.Module):
    """
    Causal Convolution block, composed of two causal convolutions
    """


    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, activation_code=0):
        super().__init__()

        activation_func = get_activation_func(activation_code)
        self.final = final
        # Compute left padding, 
        # for convolutions to be causal we need left padding, to avoid seeing the future
        padding = (kernel_size - 1) * dilation
        

        # Weight normalization 
        '''conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding= padding, dilation= dilation
        ))'''

        conv1 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding= padding, dilation= dilation
        ))

        chomp1 = Chomp(padding)
        activation1 = activation_func

        '''conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding= padding, dilation= dilation
        ))'''

        # torch.nn.utils.weight_norm
        
        conv2 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding= padding, dilation= dilation
        ))

        chomp2 = Chomp(padding)
        activation2 = activation_func

        self.causal = nn.Sequential(
            conv1, chomp1, activation1, conv2, chomp2, activation2
        )


        # Residual Connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = None

        #self.activation3 = activation_func
    
    def forward(self, x):
        out_causal = self.causal(x)

        if self.residual is None:
            res = x
        else:
            res = self.residual(x)
        
        return out_causal + res
        if self.final:
            return self.activation3(out_causal + res)
        else:
            return out_causal + res
        



class CausalCNN(nn.Module):
    """
    Causal CNN, composed of causal CNN blocks
    """
    def __init__(self, in_channels, channels, depth, out_channels, kernel_size):
        super().__init__()

        layers = []
        dilation_size = 1 # Initial dilation

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]

            dilation_size *= 2 # Doubles dilation each layer. IDK WHY

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    



class CausalCNNEncoder(nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size):
        super().__init__()

        causalCNN = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )

        reduce_size = nn.AdaptiveAvgPool1d(1)
        squeeze = SqueezeChannels()
        linear = nn.Linear(reduced_size, out_channels)

        self.network = torch.nn.Sequential(
            causalCNN, reduce_size, squeeze, linear
        )


    def forward(self, x):
        return self.network(x)
    


