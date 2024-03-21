import torch.nn as nn
import torch
from collections import OrderedDict
import random
random.seed(0)



class CAE(nn.Module):
    """
    Convolutional Autoencoder (CAE) neural network model, right now configured for input of shape (batch, channel=1, N_x=512).

    The encoder and decoder both use Conv1d. Only tunable parameter is the latent dimension.
    """

    def __init__(self,  latent_size, weight_init_name='kaiming_uniform', random_seed=0):
        """
        Initialize the CAE model.

        Args:
            latent_size (int): The size of the latent representation.
            weight_init_name (str, optional): Name of the weight initialization method.
                                              Default is 'kaiming_uniform', corresponding to He initialization. 
            random_seed (int, optional): Random seed for reproducibility. Default is 0.
        """
        super(CAE, self).__init__()
        torch.manual_seed(random_seed)
        self.latent_size = latent_size
        self.initialize_encoder()
        self.initialize_decoder()
        self.initialize_weights(weight_init_name)


    def initialize_encoder(self):
        self.encoder = nn.Sequential(OrderedDict([
            ('enc_conv0', nn.Conv1d(in_channels=1, out_channels=2, kernel_size=7, stride=2)),
            ('tanh0', nn.Tanh()),
            ('enc_conv1', nn.Conv1d(in_channels=2, out_channels=4, kernel_size=7, stride=2)),
            ('tanh1', nn.Tanh()),
            ('enc_conv2', nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=2)),
            ('tanh2', nn.Tanh()),
            ('enc_conv3', nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2)),
            ('tanh3', nn.Tanh()),
            ('enc_conv4', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)),
            ('tanh4', nn.Tanh()),
            ('enc_conv5', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)),
            ('tanh5', nn.Tanh()),
            ('enc_flat', nn.Flatten()),
            ('enc_linear_dense', nn.Linear(in_features=(64), out_features=self.latent_size)),
            ('tanh_lin', nn.Tanh())
        ]))


    def initialize_decoder(self):
        self.decoder = nn.Sequential(OrderedDict([
            ('dec_linear_dense', nn.Linear(in_features=self.latent_size, out_features=(64))),
            ('dec_tanh_dense', nn.Tanh()),
            ('dec_unflat', nn.Unflatten(1, (64, 1))),
            ('dec_tconv1', nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=0)),
            ('dec_tanh1', nn.Tanh()),
            ('dec_tconv2', nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=0)),
            ('dec_tanh2', nn.Tanh()),
            ('dec_tconv3', nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=0)),
            ('dec_tanh3', nn.Tanh()),
            ('dec_tconv4', nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=5, stride=2, padding=0)),
            ('dec_tanh4', nn.Tanh()),
            ('dec_tconv5', nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=7, stride=2, padding=1)),
            ('dec_tanh5', nn.Tanh()),
            ('dec_tcon6', nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=8, stride=2, padding=0)),
            ('dec_tanh6', nn.Tanh())
        ]))

    def initialize_weights(self, weight_init_name):
        weight_init_dict = {
            'xavier_uniform': nn.init.xavier_uniform_, 
            'xavier_normal': nn.init.xavier_normal_, 
            'kaiming_uniform': nn.init.kaiming_uniform_, # default choice of torch, corresponds to He initialization
            'kaiming_normal': nn.init.kaiming_uniform_,
        }

        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                weight_init_dict[weight_init_name](module.weight)
        for name, module in self.decoder.named_modules() or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv1d):
                weight_init_dict[weight_init_name](module.weight)


    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded