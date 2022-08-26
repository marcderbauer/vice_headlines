# Train and validation go in here, making use of the data and model
# Could implement batching data here

from pickle import FALSE, TRUE
import time
from numpy import CLIP
import wandb
import os
import torch
import torch.nn as nn
import random
from utils import randomChoice, timeSince

from data import Data
import model

LOG_WANDB = False

DATA_LOCATION = "data/titles_cleaned.txt"
SEED = None

LEARNING_RATE = 0.0005 # in wlm github it's 20??
N_EPOCHS = 1000
MAX_LENGTH = 20
HIDDEN_SIZE = 128
PRINT_EVERY = 5000
CRITERION = nn.NLLLoss()
CLIP = 0.25


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

data = Data()
data.load(DATA_LOCATION)
num_tokens = data.num_words
model = model.RNN(num_tokens, HIDDEN_SIZE, num_tokens, num_layers=None, num_categories=data.num_categories)
# TODO: look into num_layers


# One-hot vector for category
def categoryTensor(category):
    li = data.all_categories.index(category)
    tensor = torch.zeros(1, data.num_categories)
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
def randomTrainingExample():
    category, line = data.randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor



# TRAINING
def evaluate(data_source):
    model.eval()
    total_loss = 0.
    #num_tokens = data.num_words #do I really need to write this again?
    hidden = model.initHidden()
    with torch.no_grad():
        for i in range(0, len(data_source)): # Iterate over entire eval set?
            category_tensor, _ = data.randomTrainingPair() # TODO: THIS NEEDS TO BE CHANGED, WORKS ONLY CAUSE THERE'S ONLY ONE CATEGORY RN
            input_line_tensor = inputTensor(data_source[i])
            target_line_tensor = targetTensor(data_source[i])
            output, hidden = model(category_tensor, input_line_tensor[i], hidden) #  TODO: Should ilt be accessed by index here?
            #hidden = repackage_hidden(hidden)??
        total_loss += len(input_line_tensor) * CRITERION(output, target_line_tensor).item()
    return total_loss / (len(data_source) - 1 )

def train(model, category_tensor, input_line_tensor, target_line_tensor):
    """
    Single training step from the model side. 
    # TODO: Should this not be an entire epoch worth of training? Would that be better?
    """
    target_line_tensor.unsqueeze_(-1) # Arranges data in a vertical tensor
    hidden = model.initHidden() # is the hidden state re-initialized every epoch?

    model.zero_grad() # TODO: Is this related to dropout? Or what does this do?
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = CRITERION(output, target_line_tensor[i])
        loss += l

    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    # TODO: look into this

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha = LEARNING_RATE)
    
    return output, loss.item() / input_line_tensor.size(0)

# if name is main?

# TODO: surround by try/except
# ITERATE OVER EPOCHS
for epoch in range(1, N_EPOCHS+1):
    # Is term epoch here accurate? Shouldn't an epoch be the entire dataset once?
    # Currently it's a fixed amount of iterations

    # Loss is reset for every training step. Could also declare within train / eval functions
    all_losses = []
    total_loss = 0.
    start = time.time()

    example = randomTrainingExample()
    output, train_loss = train(model, *example)
    eval_loss = evaluate(data.dev)

    total_loss += train_loss

    if epoch % PRINT_EVERY == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), epoch, epoch / N_EPOCHS * 100, loss))
    
    # w and b
    if LOG_WANDB:
        wandb.log( {"train_loss": train_loss,
                    "eval_loss": eval_loss})
        wandb.watch(model)

