from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random
import time
import math

# Currently declared twice, find a way around this
all_letters = string.ascii_letters + " .,;'-"

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

# Read a file and split into lines
def readSplitLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()).split() for line in some_file]

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def split_dset(dset, train, dev, test=None):
    # TODO: Included in data class now, can be removed soon.
    """
    Takes percentage that each split is supposed to be at and splits the dataset accodringly.
    input percentages must add to one
    """
    assert train + dev + test == 1, "Percentage doesn't equal one" 
    split1 = round(train * len(dset))
    split2 = round((train+dev) * len(dset))
    tr = dset[:split1]
    if test:
        d = dset[split1:split2]
        te = dset[split2]
        return tr, d, te
    else:
        d = dset[split1:]
        return tr, d


if __name__ == "__main__":
    ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    print(split_dset(ds, 0.8, 0.1, 0.1))