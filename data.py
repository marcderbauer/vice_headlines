# load, tokenize, etc
import os
import torch
from utils import findFiles, randomChoice, readLines, split_dset

TRAIN = 0.8
DEV = 0.1
TEST = 0.1

class Data():
    def __init__(self) -> None:
        self.all_categories = []
        self.category_lines = {}
        self.all_words = []
        self.num_words = None


    def load(self, path, train=TRAIN, dev=DEV, test=TEST):
        """
        Loads a file and splits it into train, dev and test at a 80/10/10 treshold
        """
        self._load_file(path)
        self.split(train, dev, test)


    def _load_file(self, path):
        """
        Loads a file into the data class
        """
        words = []
        for filename in findFiles(path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = readLines(filename)
            lines = [line.split() for line in lines]
            self.category_lines[category] = lines
            for line in lines:
                words = [*words, *line]
        self.all_words = list(set(words)) # remove duplicates
        self.num_words = len(self.all_words) + 1 # for EOS
        # TODO: for adding SOS token it's probably + 2
        print("######### Loading file done. #########")
    

    def split(self, train=TRAIN, dev=DEV, test=TEST):
        """
        Splits the all_words
        """
        assert self.all_words, "You need to load a file before splitting the dataset."
        self.train, self.dev, self.test = split_dset(self.all_words, train, dev, test)
        print("######### Split dataset successfully. #########")
    

    def get_random_word(self):
        """
        Picks a random word from all words.
        TODO: In the future it would be great to have a start token instead!
        Also I really don't need a method for this...
        """
        return randomChoice(self.all_words)
    

    def randomTrainingPair(self):
        """
        Get a random category and random line from that category
        """
        category = randomChoice(self.all_categories)
        line = randomChoice(self.category_lines[category])
        return category, line
