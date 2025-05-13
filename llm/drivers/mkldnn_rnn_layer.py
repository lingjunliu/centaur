import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = torch.tensor(input_dict["weight"])
    bias_tensor = torch.tensor(input_dict["bias"])
    h0_tensor = torch.tensor(input_dict["h0"])
    seq_lengths = input_dict.get("seq_lengths", torch.tensor([]))
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    train = input_dict.get("train", True)
    bidirectional = input_dict.get("bidirectional", False)
    mode = input_dict.get("mode", 'LSTM')
    num_layers = input_dict.get("num_layers", 1)
    hidden_size = input_dict.get("hidden_size")
    has_biases = True

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        bias_tensor = bias_tensor.cuda()
        h0_tensor = h0_tensor.cuda()
        if len(seq_lengths) > 0:
            seq_lengths = seq_lengths.cuda()

    weights = weight_tensor
    biases = bias_tensor
    hx = h0_tensor
    cx = torch.zeros_like(h0_tensor) if mode == 'LSTM' else None
    batch_sizes = seq_lengths if len(seq_lengths) > 0 else None
    reverse = False

    if batch_sizes is None:
        batch_sizes = torch.tensor([])

    if mode == 'LSTM':
      result = torch.mkldnn_rnn_layer(input_tensor, weights, biases, hx, cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train, dropout)
    else:
      result = torch.mkldnn_rnn_layer(input_tensor, weights, biases, hx, None, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train, dropout)


    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result": result[0].numpy(), "hidden": result[1].numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight_tensor = tf.constant(input_dict["weight"])
        bias_tensor = tf.constant(input_dict["bias"])
        h0_tensor = tf.constant(input_dict["h0"])
        seq_lengths = input_dict.get("seq_lengths", tf.constant([]))
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)
        mode = input_dict.get("mode", 'LSTM')
        num_layers = input_dict.get("num_layers", 1)

        if mode == 'LSTM':
            rnn_cell = tf.keras.layers.LSTM(h0_tensor.shape[-1], return_sequences=True, return_state=True)
        elif mode == 'GRU':
            rnn_cell = tf.keras.layers.GRU(h0_tensor.shape[-1], return_sequences=True, return_state=True, activation='tanh')
        elif mode == 'RNN_TANH':
            rnn_cell = tf.keras.layers.SimpleRNN(h0_tensor.shape[-1], return_sequences=True, return_state=True, activation='tanh')
        elif mode == 'RNN_RELU':
            rnn_cell = tf.keras.layers.SimpleRNN(h0_tensor.shape[-1], return_sequences=True, return_state=True, activation='relu')
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if bidirectional:
            rnn_cell = tf.keras.layers.Bidirectional(rnn_cell)

        if batch_first:
            inputs = tf.transpose(input_tensor, perm=[1, 0, 2])
            initial_state = [tf.transpose(h0_tensor, perm=[1, 0, 2])]
            outputs, state = rnn_cell(inputs, initial_state=initial_state)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
        else:
            outputs, state = rnn_cell(input_tensor, initial_state=h0_tensor)
            
        if isinstance(state, list) or isinstance(state, tuple):
            hidden_state = state[0].numpy()
        else:
            hidden_state = state.numpy()

        return {"result": outputs.numpy(), "hidden": hidden_state}

def main():
    A_TOL = 0.01

    input_size = 10
    hidden_size = 20
    seq_length = 5
    batch_size = 2
    num_layers = 1

    input_data = {
        "input": np.random.rand(seq_length, batch_size, input_size).astype(np.float32),
        "weight": np.random.rand(num_layers * (2 if False else 1), hidden_size, input_size).astype(np.float32),
        "bias": np.random.rand(num_layers * (2 if False else 1), hidden_size).astype(np.float32),
        "h0": np.random.rand(num_layers * (2 if False else 1), batch_size, hidden_size).astype(np.float32),
        "seq_lengths": np.array([]),
        "batch_first": False,
        "dropout": 0.0,
        "train": True,
        "bidirectional": False,
        "mode": 'LSTM',
        "num_layers": num_layers,
        "hidden_size": hidden_size
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["hidden"], tf_result["hidden"], atol=A_TOL), "Hidden states do not match"

    print("Success")

if __name__ == "__main__":
    main()