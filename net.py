import torch.nn.functional as F
import torch.nn as nn
import torch
from conv_lstm import ConvLSTM


class SpatialEncoder(nn.Module):

    def __init__(self):
        super(SpatialEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels = 1, out_channels= 128, kernel_size= (1,11,11), stride=(1,4,4))
        self.conv2 = nn.Conv3d(in_channels= 128, out_channels= 64, kernel_size= (1,5,5), stride=(1,2,2))
        self.convlstm = ConvLSTM(input_dim=64,
                 hidden_dim=[64,32,64],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=True)
        self.convt1 = nn.ConvTranspose3d(64, 128,(1,5,5), stride=(1,2,2))
        self.convt2 = nn.ConvTranspose3d(128, 1,(1,11,11), stride=(1,4,4))
    def forward(self, X):

        X = self.conv1(X)
        X = F.tanh(X)
        X = self.conv2(X)
        X = F.tanh(X)
        # batch, channels, time, h, w
        assert X.size()[1:] == torch.Size([64, 10, 26, 26])
        # batch, time, channels, h, w
        X = X.permute([0, 2, 1, 3, 4])
        layer_output_list, last_state_list = self.convlstm(X)
        X = layer_output_list[-1]
        # batch, depth, channels, h, w
        assert X.shape[1:] == torch.Size([10,64,26,26])
        # batch, channels, depth, h, w
        X = X.permute([0, 2, 1, 3, 4])
        X = self.convt1(X)
        X = F.tanh(X)
        X = self.convt2(X)
        X = F.sigmoid(X)
        assert X.shape[1:] == torch.Size([1, 10,227,227])
        return X
