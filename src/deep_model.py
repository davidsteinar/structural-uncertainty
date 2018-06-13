import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, z_size, output_size, dropout_p):
        super(Model, self).__init__()
        self.dropout_p = dropout_p

        self.h1 = nn.Linear(input_size, hidden_size1)
        self.h2 = nn.Linear(hidden_size1, hidden_size2)
        self.z  = nn.Linear(hidden_size2, z_size)
        self.h4 = nn.Linear(z_size, hidden_size2)
        self.h5 = nn.Linear(hidden_size2, hidden_size1)
        self.h6 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h1(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h2(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.z(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h4(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.leaky_relu(self.h5(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.h6(x)
        return x