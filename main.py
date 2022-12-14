#!/usr/bin/env python
# To run this as a background process:
# python main.py &
# Might to manually kill the process when you're done, otherwise it will stay active in the background

# TODO:
# Could implement batching data here

import time
from numpy import CLIP
import wandb
import os
import torch
import torch.nn as nn
import random
from utils import randomChoice, timeSince
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from shutil import rmtree

from data import Data2
from model import RNN

# -----------------------------------------------------------------------------------------#
#                                           PARAMS                                         #
# -----------------------------------------------------------------------------------------#

# General Parameters
DATA_LOCATION = "data/titles_cleaned.txt"#test/file*.txt" # "data/overfit_char.txt"
MODELS_DIR = "models"
LEVEL = "word"              # whether the model operates at a "char" or "word" level
LOG_WANDB = True
SEED = None
SAVE_EVERY = 1              # Save every X epochs (aside from best model)
LOG_ITER = 50              # Log every X steps

# Model Parameters
LEARNING_RATE = 0.0005      # in wlm github it's 20??
N_EPOCHS = 100              # Number of epochs
HIDDEN_SIZE = 128           # Size of hidden Layer
CRITERION = nn.NLLLoss()    # Loss function used
CLIP = 0.25                 # Gradient clipping

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    torch.device("mps")

# -----------------------------------------------------------------------------------------#
#                                           SETUP                                          #
# -----------------------------------------------------------------------------------------#

# Set W and B logging
if LOG_WANDB:
    wandb.init(project="vice_headlines", entity="marcderbauer")

    wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": N_EPOCHS,
    "batch_size": 1
    }

# Generate and set seed
if not SEED:
    SEED = random.randrange(100000)
    print(f"No seed set. Current seed: {SEED}\n")
torch.manual_seed(SEED)

# Create directory to save models in
SAVE_DIR = os.path.join(MODELS_DIR, os.path.basename(DATA_LOCATION).split(".")[0]) # "models/names"#
if os.path.exists(SAVE_DIR):
    rmtree(SAVE_DIR)
os.mkdir(SAVE_DIR)

# Load data and split into train and test
data = Data2(DATA_LOCATION, char=False)
train_data, test_data = random_split(data, [round(len(data) * 0.8), round(len(data) * 0.2)])
#train_data = data
#test_data = data
train_dl = DataLoader(train_data, batch_size=None, shuffle=True)

test_dl = DataLoader(test_data, batch_size=None, shuffle=True)

num_tokens = data.num_words

# Load the model
# TODO: look into num_layers
model = RNN(num_tokens, HIDDEN_SIZE, num_tokens, num_layers=None, num_categories=data.num_categories)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)



# TRAINING
def evaluate(data_source):
    model.eval()
    loss = 0. 
    hidden = model.initHidden()
    with torch.no_grad(): 

        category_tensor, input_line_tensor, target_line_tensor = randomChoice(data_source)

        target_line_tensor.unsqueeze_(-1)

        for i in range(input_line_tensor.size(0)):
            output, hidden = model(category_tensor, input_line_tensor[i], hidden)
            l = CRITERION(output, target_line_tensor[i])
            loss += l
        
        return loss.item() / input_line_tensor.size(0)



def train(model, category_tensor, input_line_tensor, target_line_tensor):
    """
    Single training step from the model side. 
    """
    target_line_tensor.unsqueeze_(-1) # Arranges data in a vertical tensor
    hidden = model.initHidden() # is the hidden state re-initialized every epoch?

    optimizer.zero_grad() # TODO: Is this related to dropout? Or what does this do?
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = CRITERION(output, target_line_tensor[i]) # Fails when i = 2 -> value returned is 20073 # i=0: 31 i=1 41
        loss += l

    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    #torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    # TODO: look into this

    optimizer.step()
    #weights = model.embedding.weight.detach().clone().numpy()
    # for p_counter, p in enumerate(model.parameters()):
    #     if p.grad is not None:
    #         p.data.add_(p.grad.data, alpha = -LEARNING_RATE)
    
    return output, loss.item() / input_line_tensor.size(0)


def main():
    lr = LEARNING_RATE
# At any point you can hit Ctrl + C to break out of training early.
    try:
        best_eval_loss = None
        total_train_loss = 0.
        total_eval_loss = 0.
        start = time.time()

        for epoch in range(1, N_EPOCHS+1):
            # Is term epoch here accurate? Shouldn't an epoch be the entire dataset once?
            # Currently it's a fixed amount of iterations
            
            epoch_train_loss = 0.
            epoch_eval_loss = 0.
            train_it = iter(train_dl)

            for i, example in enumerate(train_it):#enumerate(train_data):
                _, train_loss = train(model, *example)
                eval_loss = evaluate(test_data)

                total_train_loss += train_loss
                total_eval_loss += eval_loss
                epoch_train_loss += train_loss
                epoch_eval_loss += eval_loss

                if  i % LOG_ITER == 0:
                    if LOG_WANDB:
                        wandb.log( {"train_loss": train_loss,
                                    "eval_loss": eval_loss})
                        wandb.watch(model)
                    print('%s (%d %d%%) train: %.4f   eval: %.4f' % (timeSince(start), epoch, epoch / N_EPOCHS * 100, train_loss, eval_loss))
            
            train_loss = epoch_train_loss / len(train_data)
            eval_loss = epoch_eval_loss / len(train_data)

            

            if epoch % SAVE_EVERY == 0:
                print('%s (%d %d%%) train: %.4f   eval: %.4f' % (timeSince(start), epoch, epoch / N_EPOCHS * 100, train_loss, eval_loss))
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
                best_eval_loss = eval_loss
            else:
                
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                #LEARNING_RATE /= 4.0
                lr = lr / 4.0
                #TODO: This one probably won't get updated (even without the continue statement)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if __name__ == "__main__":
    main()