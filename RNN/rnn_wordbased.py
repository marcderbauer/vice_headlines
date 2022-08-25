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
import wandb
from utils import findFiles, readLines, unicodeToAscii, randomChoice, timeSince

############ SETUP STAGE ####################

# TODO: Handle these? Should they be part of the RNN?
# Should they be part of the main class?

LOG_WANDB = True

MAX_LENGTH = 20
N_ITERS = 1000
PRINT_EVERY = 5000
PLOT_EVERY = 500
CRITERION = nn.NLLLoss()
LEARNING_RATE = 0.0005

all_categories = []
category_lines = {}

if LOG_WANDB:
    wandb.init(project="vice_headlines", entity="marcderbauer")

    wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": N_ITERS,
    "batch_size": 1
    } # TODO: Does the current RNN have a batch size?

# TODO: There should be one funtion that splits the lines and returns them as such

def load_data(path):
    # TODO: Should this be here?
    # It feels like it would make more sense as part of the RNN
    # Maybe rename the RNN to something more specific, so it doesn't feel like i need to keep it super generic?
    words = []
    for filename in findFiles(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        lines = [line.split() for line in lines]
        category_lines[category] = lines
        for line in lines:
            words = [*words, *line]
    words = list(set(words)) # remove duplicates
    return words, all_categories, category_lines


# Actually having them up here seems like a good idea
# TODO Make uppercase
ALL_LETTERS, ALL_CATEGORIES, CATEGORY_LINES = load_data('data/titles_cleaned.txt')#string.ascii_letters + " .,;'-|"#  + "0123456789" # TODO: This needs to be inependent / adjust to the load data
N_LETTERS = len(ALL_LETTERS) + 1 # Plus EOS marker

def get_random_word(vocab):
    return randomChoice(vocab) # TODO: Could instead sample from the list of first words of all video titles


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, criterion = CRITERION, learning_rate = LEARNING_RATE):
        super(RNN, self).__init__()

        self.category_lines = CATEGORY_LINES
        self.all_categories = ALL_CATEGORIES
        self.n_categories = len(self.all_categories)

        self.hidden_size = hidden_size
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.i2h = nn.Linear(self.n_categories + input_size + hidden_size, hidden_size) # TODO: Could add another hidden layer h2h here
        self.i2o = nn.Linear(self.n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)


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
    

    def load_data(self, path):
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
        # TODO: Adjust for words
        tensor = torch.zeros(len(line), 1, N_LETTERS)
        for li in range(len(line)): # TODO: This doesn't interpret words as single entities here, but instead iterates over their individual letters when generating
            letter = line[li]
            # TODO: This needs to be adjusted
            tensor[li][0][ALL_LETTERS.index(letter)] = 1
        return tensor


    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self, line):
        # TODO: Adjust for words
        letter_indexes = [ALL_LETTERS.index(line[li]) for li in range(1, len(line))] # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
        letter_indexes.append(N_LETTERS - 1) # EOS
        return torch.LongTensor(letter_indexes)
    

    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample(self):
        category, line = self.randomTrainingPair()
        category_tensor = self.categoryTensor(category)
        input_line_tensor = self.inputTensor(line)
        target_line_tensor = self.targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor
    

    def train(self, category_tensor, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1) # TODO: Not sure what this does
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

    # Sample from a category and starting letter
    def sample(self, category, start_letter='A'):
        with torch.no_grad():  # no need to track history in sampling
            category_tensor = self.categoryTensor(category)
            input = self.inputTensor([start_letter])
            hidden = self.initHidden()

            output_name = start_letter

            for i in range(MAX_LENGTH):
                output, hidden = self(category_tensor, input[0], hidden)
                topv, topi = output.topk(1)
                topi = topi[0][0] # index of top value = letter
                if topi == N_LETTERS - 1:
                    break
                else:
                    letter = ALL_LETTERS[topi]
                    output_name += " " + letter
                input = self.inputTensor([letter])

            return output_name


def training():
    rnn = RNN(N_LETTERS, 128, N_LETTERS)

    # TODO: Logging should go in here

    all_losses = []
    total_loss = 0 # Reset every plot_every iters

    start = time.time()

    for iter in range(1, N_ITERS + 1):
        output, loss = rnn.train(*rnn.randomTrainingExample())
        total_loss += loss

        if iter % PRINT_EVERY == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / N_ITERS * 100, loss))

        if iter % PLOT_EVERY == 0:
            all_losses.append(total_loss / PLOT_EVERY)
            total_loss = 0
        
        # w and b
        if LOG_WANDB:
            wandb.log({"loss": loss})
            wandb.watch(rnn)
    return rnn


# Get multiple samples from one category and multiple starting letters
def generate(rnn, category, start_letters='ABC'):
    #for start_letter in start_letters:
    print(rnn.sample(category, start_letters))

if __name__ == "__main__":
    # TODO: how to handle train and training_stage
    rnn = training()
    generate(rnn, 'titles_cleaned', get_random_word(ALL_LETTERS)) # TODO: Maybe turn word into a list here to avoid changing the sample() code
    # TODO: Save model once it reaches here
    # generate(rnn,'German', 'GER')
    # generate(rnn,'Spanish', 'SPA')
    # generate(rnn,'Chinese', 'CHI')
    
    # TODO: Next steps
    # Split into training and validation
    # -> Continually monitor validation
    # Try LSTM, Convolutions, etc.
    # TODO: Could try to set this up with pytorch lightning for monitoring, checkpointing, visualizaion etc



######################################################################
# Exercises
# =========
#
# -  Try with a different dataset of category -> line, for example:
#
#    -  Fictional series -> Character name
#    -  Part of speech -> Word
#    -  Country -> City
#
# -  Use a "start of sentence" token so that sampling can be done without
#    choosing a start letter
# -  Get better results with a bigger and/or better shaped network
#
#    -  Try the nn.LSTM and nn.GRU layers
#    -  Combine multiple of these RNNs as a higher level network
#
