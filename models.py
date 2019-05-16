import torch
import numpy as np
from functions import make_one_hot


class MyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, char_to_idx, idx_to_char, use_gpu):
        super(MyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.use_gpu = use_gpu

        # create recurrent and linear layers
        lstm = torch.nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_size)
        linear = torch.nn.Linear(self.hidden_size, self.vocab_size)

        self.lstm = lstm
        self.linear = linear
        self.reset_states()
        self.h_prev = None
        self.c_prev = None

    def forward(self, inputs):

        assert(inputs.shape[1] == 1)
        assert(inputs.shape[2] == self.vocab_size)
        seq_length = inputs.shape[0]

        h, (h_prev, c_prev) = self.lstm(inputs, (self.h_prev, self.c_prev))

        assert(h.shape == (seq_length, 1, self.hidden_size))

        out = self.linear(h)

        assert(out.shape == (seq_length, 1, self.vocab_size))
        self.h_prev = h_prev.detach()
        self.c_prev = c_prev.detach()

        return out

    def reset_states(self):
        self.h_prev = torch.zeros([1, 1, self.hidden_size])
        self.c_prev = torch.zeros([1, 1, self.hidden_size])
        if self.use_gpu:
            self.h_prev, self.c_prev = self.h_prev.cuda(), self.c_prev.cuda()

    def predict_char(self, seed_index):
        vocab_size = self.vocab_size
        one_hot_vector = make_one_hot([seed_index], vocab_size)
        inputs = one_hot_vector.reshape((1, 1, vocab_size))
        inputs = torch.tensor(inputs, dtype=torch.float)
        if self.use_gpu:
            inputs = inputs.cuda()
        out = self(inputs)
        if self.use_gpu:
            out = out.cpu()
        out = torch.nn.functional.softmax(out, dim=2)
        out = out.data.numpy()
        pred = out.reshape(vocab_size)
        pred_index = np.random.choice(range(vocab_size), p=pred)
        return pred_index

    def generate_sample(self, seed_string, size):
        self.reset_states()
        seed_string = [self.char_to_idx[i] for i in seed_string]
        sample = []
        for i in seed_string:
            index = self.predict_char(i)
        sample.append(index)

        for i in range(size - 1):
            index = self.predict_char(index)
            sample.append(index)
        ret = ''.join([self.idx_to_char[i] for i in sample])
        return ret
