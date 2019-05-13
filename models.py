import torch


class MyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(MyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # create recurrent and linear layers
        lstm = torch.nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_size)
        linear = torch.nn.Linear(self.hidden_size, self.vocab_size)

        # self.rnn = rnn
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
