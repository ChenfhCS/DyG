import torch
import torch.nn as nn

class Classifier(torch.nn.Module):
    def __init__(self, input_dim = None, hidden_dim = 100, out_feature = 2):
        super(Classifier, self).__init__()
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.Softmax(dim=1)
        self.activation3 = torch.nn.Sigmoid()

        assert input_dim, 'Unspecified number of input features'
        
        num_feats = input_dim

        self.layer_1 = torch.nn.Linear(num_feats, hidden_dim)
        self.layer_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = torch.nn.Linear(hidden_dim, out_feature)

        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)

        # init the layer
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.constant_(self.layer_1.bias, 0)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.constant_(self.layer_2.bias, 0)
        nn.init.xavier_uniform_(self.layer_3.weight)
        nn.init.constant_(self.layer_3.bias, 0)

    # complex model leads to a poor performance!
    def forward(self, x):
        # x = self.activation1(self.layer_1(x))
        # x = self.dropout(self.norm(x))

        # x = self.activation1(self.layer_2(x))
        # x = self.dropout(self.norm(x))

        # x = self.layer_3(x)
        # x = self.dropout(x)
        # x = self.activation2(x)

        x = self.layer_1(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.activation1(x)

        x = self.layer_3(x)
        x = self.dropout(x)
        x = self.activation2(x)

        return x

# class Classifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, 2)
#         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.softmax(out)
#         return out