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


def predict_char(model, seed_index):
    one_hot_vector = make_one_hot([seed_index], model.vocab_size)
    inputs = one_hot_vector.reshape((1, 1, model.vocab_size))
    inputs = torch.tensor(inputs, dtype=torch.float)
    out = model(inputs)
    out = torch.nn.functional.softmax(out, dim=2)
    out = out.data.numpy()
    pred = out.reshape(model.vocab_size)
    pred_index = np.argmax(pred)
    return pred_index


def generate_sample(model, size, seed_string):
    model.reset_states()
    sample = []
    for i in seed_string:
        index = predict_char(model, i)
    sample.append(index)

    for i in range(size - 1):
        index = predict_char(model, index)
        sample.append(index)
    return sample


def train(model, optimizer, loss_fn, num_epochs, data, seq_length, verbose=False):
    data_one_hot = make_one_hot(data, model.vocab_size)
    dtype = torch.float
    data_length = len(data)
    seqs_per_epoch = int(data_length / seq_length)  # will this work if data/seq divides exactly?

    data3d = np.expand_dims(data_one_hot, 1)
    for epoch in range(num_epochs):
        model.reset_states()

        if epoch != 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

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


def print_sample(model, seed_string, length, char_to_idx, idx_to_char):
    sample = generate_sample(model, length, [char_to_idx[i] for i in seed_string])
    sample = ''.join([idx_to_char[i] for i in sample])
    print(sample)
