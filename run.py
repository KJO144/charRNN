import torch.nn.functional as F
from functions import *
from models import *
import time
from os.path import isfile

# filename = 'input.txt'
filename = 'wodehouse_right_ho_jeeves.txt'
data_raw = open(filename, 'r').read()  # should be simple plain text file

table = str.maketrans(dict.fromkeys('ï»¿_\xa0¡§¨©ªÃ´'))
data_raw = data_raw.translate(table)

data, vocab_size, idx_to_char, char_to_idx = data_from_text(data_raw)

seq_length = len(data) // 100
hidden_size = 200
num_layers = 2
learning_rate = 0.001
num_epochs = 50
saved_model_file = 'model.pt'

use_gpu = torch.cuda.is_available()
if use_gpu:
    print('Using gpu: {}'.format(torch.cuda.get_device_name()))

model = MyLSTM(vocab_size, hidden_size, num_layers, char_to_idx, idx_to_char, use_gpu)

if isfile(saved_model_file):
    model.load_state_dict(torch.load(saved_model_file))
    print('Loading existing model: {}'.format(saved_model_file))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t_start = time.time()
train(model, optimizer, loss_fn, num_epochs, data, seq_length, False)
t_end = time.time()

torch.save(model.state_dict(), saved_model_file)

sample = model.generate_sample('m', 1000)
print(sample)
