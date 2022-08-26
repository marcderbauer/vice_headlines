# The actual RNN
# Initialises the Model and weights
# declares forward and init_hidden functions
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.5

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, num_categories, dropout=DROPOUT) -> None:
        super(RNN, self).__init__()
    
        #self.num_tokens = num_tokens
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_categories = num_categories

        self.i2h = nn.Linear(self.num_categories + input_size + hidden_size, hidden_size) # TODO: Could add another hidden layer h2h here
        self.i2o = nn.Linear(self.num_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
