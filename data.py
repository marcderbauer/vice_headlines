# load, tokenize, etc
from optparse import IndentedHelpFormatter
import os
from numpy import char
import torch
from torch.utils.data import DataLoader, Dataset
from model import GLOVE # TODO: Don't import from model
from utils import findFiles, randomChoice, readLines, split_dset
import string
from torchtext.data import get_tokenizer
import spacy
from torchtext.vocab import GloVe

TRAIN = 0.8
DEV = 0.2
TEST = 0.0
class DataLine():
    def __init__(self, content, category) -> None:
        self.content = content
        self.category = category

class Data(Dataset):
    def __init__(self, data_path, level) -> None:
        words = []
        #counter = 0
        self.data_lines = []            # List of obj: content & category -- Don't really need to have this as a class, could be a tuple
        self.category_to_lines = {}     # {category : [lines]}
        self.all_categories = []        # List of all categories # TODO Turn into dict? Is it good to use .index()?
        self.all_words = None           # List of all words
        self.num_words = None           # Number of words
        self.num_categories = None      # Number of categories
        self.lines_to_category = {}     # TODO unused


        for filename in findFiles(data_path):
            # For each file
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)

            # Read lines
            if level == "char":
                lines = [line for line in readLines(filename)]
            elif level == "word":
                lines = [line.split() for line in readLines(filename)]
            else:
                raise("Please select the correct level to operate on. Either 'char' or 'word'.")
            self.category_to_lines[category] = lines

            for line in lines:
                words = [*words, *line] # add all words in the line to words
                self.data_lines.append(DataLine(line, category))
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
        """
        One-hot encoded tensor indicating the category of the line/example
        """
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.num_categories)
        tensor[0][li] = 1
        return tensor


    def inputTensor(self, line):
        """
        One-hot matrix of first to last letters (not including EOS) for input
        """
        tensor = torch.zeros(len(line), 1, self.num_words)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_words.index(letter)] = 1 # TODO: Adjust
        return tensor # instead of last timension being one hot encoding, it should look up the correct encoding here?

        # Instead of removing it, it needs to be adjusted to return the indices of the words in GloVe
        # -> get glove index
    
    def targetTensor(self, line):
        """
        LongTensor of second letter to end (EOS) for target
        """
        letter_indexes = [self.all_words.index(line[li]) for li in range(1, len(line))] # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
        letter_indexes.append(self.num_words - 1) # EOS
        return torch.LongTensor(letter_indexes)


################################################################################
####################            DATA 2          ################################
################################################################################


class Data2(Dataset):
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
                # TODO change
                lines = [line for line in readLines(filename)]
            else:
                # TODO change
                lines = [self.tokenizer(line) for line in readLines(filename)]
            
            self.category_to_lines[category] = lines

            for line in lines:
                #words = [*words, *line] # add all words in the line to words
                line = [word.lower() for word in line] #TODO: Shitty quickfix? Or maybe put processing here?
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
        # THis above is wrong. The input line tensor is a one hot encoding of the words in the line
        # Instead of removing it, it needs to be adjusted to return the indices of the words in GloVe
        # TODO: how does this work with the embedding layer?
        # -> Depends on the shape, right?
        # If it's just a "list" of word indices then it should be fairly easy to make this work in the layer
        # The layer takes indices anyway and then returns the corresponding embedding
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
        Dense matrix of first to last letters (not including EOS) for input
        """
        tensor = torch.zeros(len(line), self.num_words, dtype=torch.int32) # Removed extra dimension here
        for li in range(len(line)):
            letter = line[li]
            tensor[li][self.get_index(letter)] = 1 # Removed extra dimension here
        return tensor # instead of last timension being one hot encoding, it should look up the correct encoding here?

        # Instead of removing it, it needs to be adjusted to return the indices of the words in GloVe
        # -> get glove index
    
    def targetTensor(self, line, char = False): # TODO: Should char be handled here?
        """
        LongTensor of second letter to end (EOS) for target
        """
        letter_indexes = [self.get_index(line[li]) for li in range(1, len(line))] # indexes of remaining letters in name (from 2nd letter onwards) + <EOS>
        letter_indexes.append(self.EOS)
        return torch.LongTensor(letter_indexes)

        # def vectorize_batch(batch):
        #     Y, X = list(zip(*batch))
        #     X = [self.tokenizer(x) for x in X]
        #     X = [tokens + [" "] * (max_words - len(tokens)) if len(tokens) < max_words else tokens[:max_words] for tokens in X]
        #     X_tensor = torch.zeros(len(batch), self.max_words, self.embed_len)
        #     for i, tokens in enumerate(X):
        #         X_tensor[i]  = self.global_vectors.get_vecs_by_token(tokens)
        #     return X_tensor.reshape(len(batch), -1), torch.tensor(Y)
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
    dataset = Data2("data/titles_cleaned.txt", char=False)#names/*.txt")
    #dataset.targetTensor("Greensmith")
    dataset.__getitem__(4)
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


