import numpy as np
import torch

def data_from_text(data_raw):
    chars = list(set(data_raw))
    chars.sort()

    vocab_size = len(chars)
    print('Data has length {} and consist of {} unique characters.'.format(len(data_raw), vocab_size))
    ch_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [ch_to_idx[ch] for ch in data_raw]
    return data, vocab_size, idx_to_char, ch_to_idx


def make_one_hot(data, vocab_size):
    one_hot = np.eye(vocab_size)[data]
    return one_hot




def train(model, optimizer, loss_fn, num_epochs, data, seq_length, verbose=False):
    data_one_hot = make_one_hot(data, model.vocab_size)
    dtype = torch.float
    data_length = len(data)
    seqs_per_epoch = int(data_length / seq_length)  # will this work if data/seq divides exactly?
    f = open("output.txt", "w+")
    data3d = np.expand_dims(data_one_hot, 1)
    for epoch in range(num_epochs):
        model.reset_states()

        if epoch != 0:
            info = 'epoch: {}, loss: {}'.format(epoch, loss)
            print(info)
            sample = model.generate_sample('a', 100)
            f.write(info + "\n\n")
            f.write(sample + "\n\n")
            f.flush()

        for i in range(seqs_per_epoch):
            start = i * seq_length
            end = i * seq_length + seq_length
            end = min(end, data_length - 1)

            inputs_raw = data3d[start:end]
            targets_raw = data[start + 1:end + 1]

            # both inputs and targets are (seq_length, 1, vocab_size)
            inputs = torch.tensor(inputs_raw, dtype=dtype)
            targets = torch.tensor(targets_raw, dtype=torch.long)

            out = model(inputs)

            out2 = out.view((seq_length, model.vocab_size))
            loss = loss_fn(out2, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print(i, loss.item())



