import itertools
import torch
from torch.utils.data import Dataset


# file_path = 'solutions_1k.txt'

# print(f'Loading {file_path} dataset...')

# # Load data from file_path
# with open(file_path, 'r', encoding='utf-8') as f:
#     mydata = [line.strip().split() for line in f]




class CharDataset(Dataset):
    def __init__(self, data):
        
        chars = sorted(list(set(list(itertools.chain.from_iterable(data)))) + ["-100"])
        data_size, vocab_size = len(data), len(chars)  # -100 sorted to the front
        max_len = max([len(data[_]) for _ in range(len(data))])
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))
        
        # TODO: Perhaps set '-100' at end of chars instead of the front.

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = max_len - 1  # for autoregressive training
        self.vocab_size = vocab_size
        self.data = data

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]

        if len(chunk) != self.max_len:
            raise ValueError('Sequence length is shorter than block_size + 1. For TSP this should not happen.')
            chunk += [-100, ] * (self.max_len - len(chunk))  # -100 can be ignored in CE
        
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]

        # Othello GPT code
        # x = torch.tensor(dix[:-1], dtype=torch.long)
        # y = torch.tensor(dix[1:], dtype=torch.long)
        # return x, y

        return dix
    
    def decode_sequence(self, sequence):
        """
        Decode a sequence of integers back into a string.
        """
        return ' ' + ' '.join([self.itos[i] for i in sequence]) + '\n'


# my_dataset = CharDataset(mydata)
# print(len(my_dataset))
# print(my_dataset[0])
# print('Done!')