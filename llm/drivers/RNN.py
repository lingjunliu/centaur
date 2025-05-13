import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["hidden"])
    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    nonlinearity = input_dict.get("nonlinearity", 'tanh')
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()

    rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    result, hn = rnn(input_tensor, hidden_state)

    if not cpu:
        result = result.cpu()
        hn = hn.cpu()

    return {"result": result.detach().numpy(), "hn": hn.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    hidden_state = tf.constant(input_dict["hidden"], dtype=tf.float32)
    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    nonlinearity = input_dict.get("nonlinearity", 'tanh')
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if nonlinearity == 'tanh':
            activation = 'tanh'
        elif nonlinearity == 'relu':
            activation = 'relu'
        else:
            raise ValueError("Nonlinearity not supported.")

        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

        rnn_cells = []
        for _ in range(num_layers):
            rnn_cells.append(tf.keras.layers.SimpleRNNCell(hidden_size, activation=activation, use_bias=bias, dropout=dropout))

        if len(rnn_cells) > 1:
            stacked_rnn_cells = tf.keras.layers.StackedRNNCells(rnn_cells)
            rnn = tf.keras.layers.RNN(stacked_rnn_cells, return_sequences=True, return_state=True)
            initial_state = tf.split(hidden_state, num_layers, axis=0)
        else:
            rnn = tf.keras.layers.RNN(rnn_cells[0], return_sequences=True, return_state=True)
            initial_state = tf.split(hidden_state, num_layers, axis=0)

        output, last_states = rnn(input_tensor, initial_state=initial_state)

        if batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])

        if num_layers > 1:
            hn = tf.concat(last_states, axis=0)
        else:
            hn = tf.concat(last_states, axis=0)

        output = output.numpy()
        hn = hn.numpy()
    
    return {"result": output, "hn": hn}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "hidden": np.random.rand(1, 3, 20).astype(np.float32),
        "input_size": 10,
        "hidden_size": 20,
        "num_layers": 1,
        "nonlinearity": 'tanh',
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match (output)"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Results do not match (hn)"

    print("Success")

if __name__ == "__main__":
    main()