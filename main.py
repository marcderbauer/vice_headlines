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
from torch.utils.data import random_split

from data import Data
import model

LOG_WANDB = True

DATA_LOCATION = "data/titles_cleaned.txt"
MODELS_DIR = "models"
SAVE_DIR = os.path.join(MODELS_DIR, os.path.basename(DATA_LOCATION).split(".")[0])
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

SEED = None

LEARNING_RATE = 0.0005 # in wlm github it's 20??
N_EPOCHS = 1000
MAX_LENGTH = 20
HIDDEN_SIZE = 128
SAVE_EVERY = 10
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
    print(f"No seed set. Current seed: {SEED}\n")
torch.manual_seed(SEED)


data = Data("data/titles_cleaned.txt")
train_data, test_data = random_split(data, [int(len(data) * 0.8), int(len(data) * 0.2)])

num_tokens = data.num_words
model = model.RNN(num_tokens, HIDDEN_SIZE, num_tokens, num_layers=None, num_categories=data.num_categories)
# TODO: look into num_layers

# Iterator over eval data. Used to take a single example for validation each iteration
eval_iter = iter(test_data)


# TRAINING
def evaluate(data_source):
    model.eval()
    loss = 0. 
    hidden = model.initHidden()
    with torch.no_grad(): 

        category_tensor, input_line_tensor, target_line_tensor = data_source.__next__() #TODO data_source.next() is broken, this is a quickfix

        target_line_tensor.unsqueeze_(-1)

        for i in range(input_line_tensor.size(0)):
            output, hidden = model(category_tensor, input_line_tensor[i], hidden)
            l = CRITERION(output, target_line_tensor[i])
            loss += l
        
        return loss.item() / input_line_tensor.size(0)



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


# At any point you can hit Ctrl + C to break out of training early.
try:
    best_eval_loss = None
    for epoch in range(1, N_EPOCHS+1):
        # Is term epoch here accurate? Shouldn't an epoch be the entire dataset once?
        # Currently it's a fixed amount of iterations

        # Loss is reset for every training step. Could also declare within train / eval functions
        total_train_loss = 0.
        total_eval_loss = 0.
        start = time.time()

        for i, example in enumerate(data):
                
            output, train_loss = train(model, *example)
            eval_loss = evaluate(eval_iter)

            total_train_loss += train_loss
            total_eval_loss += eval_loss


        if epoch % SAVE_EVERY == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), epoch, epoch / N_EPOCHS * 100, train_loss))
            with open(os.path.join(SAVE_DIR, f"{epoch}.pt"), 'wb') as f:
                torch.save(model, f)
        
        # w and b
        if LOG_WANDB:
            wandb.log( {"train_loss": train_loss,
                        "eval_loss": eval_loss})
            wandb.watch(model)

        # Save best model
        if not best_eval_loss or eval_loss < best_eval_loss:
            with open(os.path.join(SAVE_DIR, "best.pt"), 'wb') as f:
                torch.save(model, f)
            best_val_loss = eval_loss
        else:
            pass
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #LEARNING_RATE /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')