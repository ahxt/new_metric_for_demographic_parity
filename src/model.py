import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, n_features, n_hidden=128, p_dropout=0.2):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_hidden)
        self.lin4 = nn.Linear(n_hidden, 1)


    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin3(x)
        h = F.relu( x )
        x = F.dropout(x, training=self.training)

        x = self.lin4(h)
        x = torch.sigmoid(x)

        return x