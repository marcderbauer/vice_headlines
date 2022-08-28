# load, tokenize, etc
import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils import findFiles, randomChoice, readLines, split_dset
import string

TRAIN = 0.8
DEV = 0.2
TEST = 0.0
class data_line():
    def __init__(self, content, category) -> None:
        self.content = content
        self.category = category

class Data(Dataset):
    def __init__(self, data_path, level="word") -> None:
        words = []
        #counter = 0
        self.data_lines = []
        self.category_to_lines = {}
        self.all_categories = []
        self.lines_to_category = {}

        for filename in findFiles(data_path):
            # For each file
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)

            # Read lines
            if level=="char":
                lines = [line for line in readLines(filename)]
            else:
                lines = [line.split() for line in readLines(filename)]
            self.category_to_lines[category] = lines

            for line in lines:
                words = [*words, *line] # add all words in the line to words
                self.data_lines.append(data_line(line, category))
                #counter += 1

        self.all_words = list(sorted(set(words)))
        self.num_words = len(self.all_words) + 1 # for <EOS>
        self.num_categories = len(self.all_categories)
       # self.lines_category = self._invert_dict(self.category_lines_all)
    
    def __len__(self):
        return len(self.data_lines) #len(self.all_words) + 1 # for <EOS> #len(self.data_lines)

    
    def __getitem__(self, index):#-> T_co:
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
        tensor = torch.zeros(len(line), 1, self.num_words)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_words.index(letter)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self, line):
        letter_indexes = [self.all_words.index(line[li]) for li in range(1, len(line))] # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
        letter_indexes.append(self.num_words - 1) # EOS
        return torch.LongTensor(letter_indexes)


if __name__ == "__main__":
    dataset = Data("data/titles_cleaned.txt")#names/*.txt")
    dl = DataLoader(dataset=dataset, batch_size=None, shuffle=True, num_workers=0)
    # TODO: Due to unequal line length, batches don't work.
    #       To solve this we would need some padding / truncation
    # print(dataset.__getitem__(200))
    # print(dataset.__len__())

    #TODO: This fails still
    # for batch in dl:
    #     pass
    dataiter = iter(dl)
    data = dataiter.next() # TODO
    category, input_v, target = data

    # dataiter._dataset.data_lines[dataiter._sampler_iter.__next__()]