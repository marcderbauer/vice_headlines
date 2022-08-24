# based on the official pytorch tutorial on char_rnn_generation
# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
# import utils as u
import string
import os
import torch
import torch.nn as nn
import random
import time
import math
from utils import findFiles, readLines, unicodeToAscii, randomChoice, timeSince

############ SETUP STAGE ####################

# TODO: Handle these? Should they be part of the RNN?
# Should they be part of the main class?
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
MAX_LENGTH = 20


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters) + 1 # Plus EOS marker
        self.category_lines = {}
        self.all_categories = []
        self.load()
        self.n_categories = len(self.all_categories)

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(self.n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(self.n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.NLLLoss()
        self.learning_rate = 0.0005

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def load(self, path = 'data/names/*.txt'):
        for filename in findFiles(path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = readLines(filename)
            self.category_lines[category] = lines
    
    # Get a random category and random line from that category
    def randomTrainingPair(self):
        category = randomChoice(self.all_categories)
        line = randomChoice(self.category_lines[category])
        return category, line
    
    # One-hot vector for category
    def categoryTensor(self, category):
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self, line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][all_letters.find(letter)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self, line):
        letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))] # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
        letter_indexes.append(n_letters - 1) # EOS
        return torch.LongTensor(letter_indexes)
    
    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample(self):
        category, line = self.randomTrainingPair()
        category_tensor = self.categoryTensor(category)
        input_line_tensor = self.inputTensor(line)
        target_line_tensor = self.targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor
    
    def train(self, category_tensor, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        hidden = self.initHidden()

        self.zero_grad()

        loss = 0

        for i in range(input_line_tensor.size(0)):
            output, hidden = self(category_tensor, input_line_tensor[i], hidden)
            l = self.criterion(output, target_line_tensor[i])
            loss += l

        loss.backward()

        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-self.learning_rate)

        return output, loss.item() / input_line_tensor.size(0)

def training_stage():
    rnn = RNN(n_letters, 128, n_letters)

    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = rnn.train(*rnn.randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
    return rnn

# Sample from a category and starting letter
def sample(rnn, category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = rnn.categoryTensor(category)
        input = rnn.inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(MAX_LENGTH):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0] # index of top value = letter
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = rnn.inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def generate(rnn, category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(rnn, category, start_letter))

if __name__ == "__main__":
    rnn = training_stage()
    generate(rnn, 'Russian', 'RUS')
    generate(rnn,'German', 'GER')
    generate(rnn,'Spanish', 'SPA')
    generate(rnn,'Chinese', 'CHI')
        