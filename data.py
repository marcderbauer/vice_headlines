# load, tokenize, etc
import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils import findFiles, readLines
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

TRAIN = 0.8
DEV = 0.2
TEST = 0.0
GLOVE = GloVe(name="6B", dim=50)

class DataLine():
    def __init__(self, content, category) -> None:
        self.content = content
        self.category = category

class Data(Dataset):
    def __init__(self, data_path, char=False, max_words = 25, glove=GLOVE) -> None:
        self.data_lines = []            # List of obj: content & category -- Don't really need to have this as a class, could be a tuple
        self.category_to_lines = {}
        self.all_categories = []
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.max_words = max_words
        self.glove = glove
        self.num_words = len(glove) + 3 # +3 for additional tags
        self.SOS = len(glove)
        self.EOS = len(glove) + 1
        self.UNK = len(glove) + 2

        for filename in findFiles(data_path):
            # For each file
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)

            # Read lines
            if char == True:
                # TODO Move lowercasing here
                lines = [line for line in readLines(filename)]
            else:
                # TODO Move lowercasing here
                lines = [self.tokenizer(line) for line in readLines(filename)]
            
            self.category_to_lines[category] = lines

            for line in lines:
                #words = [*words, *line] # add all words in the line to words
                line = [word.lower() for word in line] #TODO: Shitty quickfix? Or maybe put processing here?
                # -> lines are all uppercase (as in self.category_to_lines...)
                self.data_lines.append(DataLine(line, category))
                

        # self.all_words = list(sorted(set(words)))
        # self.num_words = len(self.all_words) + 1 # for <EOS>
        self.num_categories = len(self.all_categories)


    def __len__(self):
        return len(self.data_lines)

    
    def __getitem__(self, index):#-> T_co:
        item = self.data_lines[index] # TODO: replace with other structure
        category_tensor = self.categoryTensor(item.category)
        input_line_tensor = self.inputTensor(item.content) # TODO: not a tensor anymore ->
        target_line_tensor = self.targetTensor(item.content)

        return (
            category_tensor, 
            input_line_tensor, 
            target_line_tensor
        )

    def get_index(self, word):
        """
        Returns the glove index of the word if it exists. If it doesn't then it returns the unknown word index
        """
        try:
            index = self.glove.stoi[word]
            return index
        except KeyError:
            return self.UNK


    def categoryTensor(self, category):
        """
        One-hot encoded tensor indicating the category of the line/example
        """
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.num_categories)
        tensor[0][li] = 1
        return tensor


    def inputTensor(self, line):
        """
        Returns a tensor with the indices corresponding to the vectors of GloVe embeddings
        """
        tensor = torch.zeros(len(line), 1, dtype=torch.int)
        for li in range(len(line)):
            word = line[li]
            tensor[li][0] = self.get_index(word)
        return tensor

    
    def targetTensor(self, line):
        """
        LongTensor of second letter to end (EOS) for target
        """
        # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
        letter_indexes = [self.get_index(line[li]) for li in range(1, len(line))]
        letter_indexes.append(self.EOS)
        return torch.LongTensor(letter_indexes)


    def decode_vector(self, vector):
        """
        Takes a list of glove indices as input and returns a list of the corresponding words.
        """
        words = []
        for index in vector:
            if index == self.EOS:
                words.append("<EOS>")
            elif index == self.SOS:
                words.append("<SOS>")
            elif index == self.UNK:
                words.append("<UNK>")
            else:
                words.append = self.glove.itos[index]
        return words


 

if __name__ == "__main__":
    dataset = Data("data/titles_cleaned.txt", char=False)#names/*.txt")
    #dataset.targetTensor("Greensmith")
    dataset.inputTensor2(['Can', 'You', 'Be', 'In', 'Love', 'With', 'Two', 'People', '|', 'Kevin', 'Gates', 'Helpline'])
    dataset.__getitem__(4)
    dl = DataLoader(dataset=dataset, batch_size=None, shuffle=True, num_workers=0)


