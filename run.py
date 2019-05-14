import torch
import torch.nn.functional as F
import numpy as np
from functions import *
from models import *
import time

filename = 'input.txt'
#filename = 'wodehouse_right_ho_jeeves.txt'
data_raw = open(filename, 'r').read()  # should be simple plain text file
#data_raw = data_raw[0:1009]

table = str.maketrans(dict.fromkeys('ï»¿_\xa0¡§¨©ªÃ´'))
data_raw = data_raw.translate(table)

data, vocab_size, idx_to_char, char_to_idx = data_from_text(data_raw)

seq_length = 250  # this is how much of the data we sample before updating the params
hidden_size = 100  # size of the hidden state vector
learning_rate = 0.01
num_epochs = 50

use_gpu = torch.cuda.is_available()
use_gpu = True

model = MyLSTM(vocab_size, hidden_size, char_to_idx, idx_to_char, use_gpu)

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t_start = time.time()
train(model, optimizer, loss_fn, num_epochs, data, seq_length, False)
t_end = time.time()

sample = model.generate_sample('m', 1000)
print(sample)

print('\n\nelapsed time: {:2}s'.format(t_end-t_start))