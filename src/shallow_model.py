import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, z_size, output_size, dropout_p):
        super(Model, self).__init__()
        self.dropout_p = dropout_p

        self.h1 = nn.Linear(input_size, hidden_size)
        self.z  = nn.Linear(hidden_size, z_size)
        self.h2 = nn.Linear(z_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.relu(self.h1(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.relu(self.z(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.relu(self.h2(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.h3(x)
        return x