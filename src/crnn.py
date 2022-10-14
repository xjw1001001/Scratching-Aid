import Resnet2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math


class ModelBase(nn.Module):
    """ Base models for all models.

    """

    def __init__(self, args):
        """ Initialize the hyperparameters of model.

        Parameters
        ----------
        args: arguments for initializing the model.

        """

        super(ModelBase, self).__init__()

        self.epsilon = 1e-4
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.attention_size = args.attention_size
        self.attention_head = args.attention_head
        self.norm_shape = [args.embedding_size]
        self.ffn_num_input = args.ffn_num_input
        self.ffn_num_hiddens = args.ffn_num_hiddens

        self.dropout_rate = args.dropout_rate

        self.num_layers = args.num_layers
        self.num_classes = args.num_classes




class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=256, drop_prob=0.3, bn_momentum=0.01):
        super(CNNEncoder, self).__init__()

        self.cnn_out_dim = cnn_out_dim
        self.drop_prob = drop_prob
        self.bn_momentum = bn_momentum

        pretrained_cnn = Resnet2D.resnet18(pretrained=False)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        self.cnn = nn.Sequential(*cnn_layers)
        self.fc = nn.Sequential(
            *[
                self._build_fc(pretrained_cnn.fc.in_features, 256, True),
                nn.ReLU(),
                nn.Dropout(p=self.drop_prob),
                self._build_fc(256, self.cnn_out_dim, False)
            ]
        )

    def _build_fc(self, in_features, out_features, with_bn=True):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, momentum=self.bn_momentum)
        ) if with_bn else nn.Linear(in_features, out_features)

    def forward(self, x_3d):
        '''
        '''
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)

            x = self.fc(x)

            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        return cnn_embedding_out

class RNNDecoder(nn.Module):
    def __init__(self, use_gru=True, cnn_out_dim=256, rnn_hidden_layers=3, rnn_hidden_nodes=256,
            num_classes=2, drop_prob=0.3, bidirectional = True):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes
        self.bidirectional  = bidirectional
        self.drop_prob = drop_prob
        self.num_classes = num_classes 

        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True,
            'bidirectional':self.bidirectional
        }

        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        if self.bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),
                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn_hidden_nodes*2, 256),
                nn.ReLU(),
                nn.Dropout(self.drop_prob),
                nn.Linear(256, self.num_classes)
            )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)
        # rnn_out shape: (batch, timestep, output_size)
        # h_n and h_c shape: (n_layers, batch, hidden_size)

        x = self.fc(rnn_out[:, -1, :]) 

        return x

if __name__ == "__main__":

    from torchsummary import summary

    cnn_encoder_params = {
        'cnn_out_dim': 128,
        'drop_prob': 0.15,
        'bn_momentum': 0.01
    }

    rnn_decoder_params = {
        'use_gru': True,
        'cnn_out_dim': 128,
        'rnn_hidden_layers': 2,
        'rnn_hidden_nodes': 256,
        'num_classes': 2,
        'drop_prob': 0.15,
        'bidirectional': True
    }

    model = nn.Sequential(
        CNNEncoder(**cnn_encoder_params),
        RNNDecoder(**rnn_decoder_params)
        # crnn.GIT2(args)
    )
    summary(model, input_size=(1,1,256,256), batch_size=16,device="cpu")
