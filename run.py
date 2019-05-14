import torch
import torch.nn.functional as F
import numpy as np
from functions import *
from models import *

filename = 'input.txt'
filename = 'wodehouse_right_ho_jeeves.txt'
data_raw = open(filename, 'r').read()  # should be simple plain text file
#data_raw = data_raw[0:1009]

table = str.maketrans(dict.fromkeys('ï»¿_\xa0¡§¨©ªÃ´'))
data_raw = data_raw.translate(table)

data, vocab_size, idx_to_char, char_to_idx = data_from_text(data_raw)

seq_length = 250  # this is how much of the data we sample before updating the params
hidden_size = 100  # size of the hidden state vector
learning_rate = 0.01
num_epochs = 50

model = MyLSTM(vocab_size, hidden_size, char_to_idx, idx_to_char)

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, optimizer, loss_fn, num_epochs, data, seq_length)

sample = model.generate_sample('m', 1000)
print(sample)