# Train and validation go in here, making use of the data and model
# Could implement batching data here

import time
import wandb
import os
import torch
import torch.nn as nn
import random
from utils import randomChoice

from data import Data
import model

LOG_WANDB = False

DATA_LOCATION = "data/titles_cleaned.txt"
SEED = None

LEARNING_RATE = 0.0005 # in wlm github it's 20??
N_EPOCHS = 1000
MAX_LENGTH = 20
HIDDEN_SIZE = 128
CRITERION = nn.NLLLoss()


if LOG_WANDB:
    wandb.init(project="vice_headlines", entity="marcderbauer")

    wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": N_EPOCHS,
    "batch_size": 1
    } # TODO: Does the current RNN have a batch size?

# SETUP
if not SEED:
    SEED = random.randrange(10000)
torch.manual_seed(SEED)

data = Data().load(DATA_LOCATION)
num_tokens = data.num_words
model = model.RNN(num_tokens, HIDDEN_SIZE, num_tokens, num_layers=None)
# TODO: look into num_layers


# One-hot vector for category
def categoryTensor(category):
    li = model.all_categories.index(category)
    tensor = torch.zeros(1, model.n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    # TODO: Adjust for words
    tensor = torch.zeros(len(line), 1, num_tokens)
    for li in range(len(line)): # TODO: This doesn't interpret words as single entities here, but instead iterates over their individual letters when generating
        letter = line[li]
        # TODO: This needs to be adjusted
        tensor[li][0][data.all_words.index(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    # TODO: Adjust for words
    letter_indexes = [data.all_words.index(line[li]) for li in range(1, len(line))] # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
    letter_indexes.append(num_tokens - 1) # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample(model):
    category, line = data.randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor



# TRAINING
def evaluate(data_source):
    model.eval()
    total_loss = 0.
    num_tokens = data.num_words #do I really need to write this again?
    hidden = model.initHidden()
    with torch.no_grad:
        pass
        # TODO: Implement

def train(model, category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1) # TODO: Not sure what this does
    hidden = model.initHidden()

    model.zero_grad() # TODO: Is this related to dropout? Or what does this do?
    

# if name is main?

# ITERATE OVER EPOCHS
for epoch in range(1, N_EPOCHS+1):
    all_lossses = []
    total_loss = 0.
    start = time.time()
    start = time.time()

    example = randomTrainingExample()
    output, train_loss = train(example)
    val_loss = evaluate(data.dev)

