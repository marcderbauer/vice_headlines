# load, tokenize, etc
import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils import findFiles, randomChoice, readLines, split_dset
import string

TRAIN = 0.8
DEV = 0.2
TEST = 0.0

class Data():
    def __init__(self) -> None:
        self.all_categories = []
        self.category_lines = {}
        self.all_words = []
        self.num_words = None
        self.num_categories = None


    def load(self, path, train=TRAIN, dev=DEV, test=TEST):
        """
        Loads a file and splits it into train, dev and test at a 80/20/00 treshold (no need for test for generative models)
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
        self.num_categories = len(self.all_categories)
        # TODO: for adding SOS token it's probably + 2
        print("######### Loading file done. #########")
    

    def split(self, train=TRAIN, dev=DEV, test=TEST):
        """
        Splits the all_words
        """
        assert self.all_words, "You need to load a file before splitting the dataset."
        if test:
            self.train, self.dev, self.test = split_dset(self.all_words, train, dev, test)
        else:
            self.train, self.dev = split_dset(self.all_words, train, dev)
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

# TODO: Need to rewrite load data to allow for train dev (test)
# Currenty self.dev for example has to relation to category
# Would be good to have self.category_lines train / dev
# Probably makes sense to test this with words



class Data2():
    def __init__(self) -> None:
        self.all_categories = []
        self.category_lines_all = {}
        self.all_words = []
        self.num_words = None
        self.num_categories = None
    
    def setup(self, path, split=0.8):
        self.load(path, split)
        self.split()
    
    def load(self, path):
        words = []
        for filename in findFiles(path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = [line.split() for line in readLines(filename)]
            self.category_lines_all[category] = lines
            for line in lines:
                words = [*words, *line]
                
        self.all_words = list(set(words)) # remove duplicates
        self.num_words = len(self.all_words) + 1 # for EOS
        self.num_categories = len(self.all_categories)
        # TODO: for adding SOS token it's probably + 2
        print("######### Loading file done. #########")





###########################################################
######## Can probably delete everything above here ########
###########################################################


class data_line():
    def __init__(self, content, category) -> None:
        self.content = content
        self.category = category

class title_dataset(Dataset):
    def __init__(self, data_path) -> None:
        words = []
        #counter = 0
        self.data_lines = []
        self.category_lines_all = {}
        self.all_categories = []
        self.lines_category = {}

        for filename in findFiles(data_path):
            # For each file
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)

            # Read lines
            lines = [line.split() for line in readLines(filename)]
            self.category_lines_all[category] = lines

            for line in lines:
                words = [*words, *line] # add all words in the line to words
                self.data_lines.append(data_line(line, category))
                #counter += 1

        self.all_words = list(set(words))
        self.num_words = len(self.all_words) + 1 # for <EOS>
        self.num_categories = len(self.all_categories)
       # self.lines_category = self._invert_dict(self.category_lines_all)
    
    def __len__(self):
        return len(self.data_lines) #len(self.all_words) + 1 # for <EOS> #len(self.data_lines)

    
    def __getitem__(self, index):#-> T_co:
        #return self.all_words[index]
        # I think what I'm trying to do here is the randomTrainingExample,
        # but instead of being random, it's based on an index
        item = self.data_lines[index]
        category_tensor = self.categoryTensor(item.category)
        input_line_tensor = self.inputTensor(item.content)
        target_line_tensor = self.targetTensor(item.content)

        return (
            category_tensor, 
            input_line_tensor, 
            target_line_tensor
        )
    

    def categoryTensor(self, category):
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.num_categories)
        tensor[0][li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self, line):
        # TODO: Adjust for words
        tensor = torch.zeros(len(line), 1, self.num_words)
        for li in range(len(line)): # TODO: This doesn't interpret words as single entities here, but instead iterates over their individual letters when generating
            letter = line[li]
            # TODO: This needs to be adjusted
            tensor[li][0][self.all_words.index(letter)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self, line):
        # TODO: Adjust for words
        letter_indexes = [self.all_words.index(line[li]) for li in range(1, len(line))] # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
        letter_indexes.append(len(self) - 1) # EOS
        return torch.LongTensor(letter_indexes)


if __name__ == "__main__":
    dataset = title_dataset("data/titles_cleaned.txt")#names/*.txt")
    dl = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    # TODO: Due to unequal line length, batches don't work.
    #       To solve this we would need some padding / truncation
    # print(dataset.__getitem__(200))
    # print(dataset.__len__())

    for batch in dl:
        pass
    dataiter = iter(dl)
    data = dataiter.next() # TODO
    category, input_v, target = data