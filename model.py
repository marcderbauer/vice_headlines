# The actual RNN
# Initialises the Model and weights
# declares forward and init_hidden functions
import math
import torch
import torch.nn as nn
from torchtext.vocab import GloVe

GLOVE = GloVe(name="6B", dim=50)
DROPOUT = 0.1

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, num_categories, dropout=DROPOUT, glove = GLOVE) -> None:
        super(RNN, self).__init__()
    
        #self.num_tokens = num_tokens
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_categories = num_categories

        ##############################
        # TODO Remove embedding layer
        ##############################
        # Worse performance could also come from the data not being linearly seperable anymore? Experiment with other layers?


        g = glove.vectors
        z = torch.zeros(3,50)
        combined = torch.cat((g,z),0)
        self.embedding = nn.Embedding.from_pretrained(combined)
        self.i2h = nn.Linear(self.num_categories + glove.dim + hidden_size, hidden_size) # TODO: Could add another hidden layer h2h here
        self.i2o = nn.Linear(self.num_categories + glove.dim + hidden_size, glove.dim)
        self.o2o = nn.Linear(hidden_size + glove.dim, len(glove)+3) #
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def forward(self, category, word, hidden):
        embedded = self.embedding(torch.nonzero(word))[0] # TODO: Simplify the data structure. Currently encoding it as vector just to decode it again right away...
        input_combined = torch.cat((category, embedded, hidden), 1)
        # The embedding for [0][x] is the same for every x besides 0
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
