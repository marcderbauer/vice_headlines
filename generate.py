# Generate new sentences sampled from the language model
# Are any of the functions here used for validation? => How dependent is this on training?
# Or is this just as a way to easily generate sentences after training?

import torch
from data import Data
import random

# TODO:
# There's a weird error in the generated output. It's all like this...
# That was after only 11 epochs. Going to let the training run for a bit more
""""
Paaa
Aaaa
Caaa
Haaa
Iaaa
"""
# -----------------------------------------------------------------------------------------#
#                                           PARAMS                                         #
# -----------------------------------------------------------------------------------------#

MAX_LENGTH = 20 # TODO: How is this handled in wlm_generate.py?
SEED = None
CHECKPOINT = "models/names/best.pt"
DATA_LOCATION = "data/names/*.txt" # "data/overfit_char.txt"
CUDA = None
LEVEL = "char"

# -----------------------------------------------------------------------------------------#
#                                           SETUP                                          #
# -----------------------------------------------------------------------------------------#

# Generate and set seed
if not SEED:
    SEED = random.randrange(100000)
    print(f"No seed set. Current seed: {SEED}\n")
torch.manual_seed(SEED)

# Cuda setup # TODO: This also needs to be done in main
if torch.cuda.is_available():
    if not CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if CUDA else "cpu")

# Load model
with open(CHECKPOINT, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

# Load data
data = Data(DATA_LOCATION, level="char")

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = data.categoryTensor(category)
        input = data.inputTensor(start_letter)
        hidden = model.initHidden()

        output_name = start_letter

        for i in range(MAX_LENGTH):
            output, hidden = model(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == data.num_words - 1:
                break
            else:
                letter = data.all_words[topi]
                output_name += letter
            input = data.inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')