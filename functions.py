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
