import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_size = input_dict["hidden_size"]
    input_size = input_dict["input_size"]

    num_layers = input_dict.get("num_layers", 1)
    nonlinearity = input_dict.get("nonlinearity", 'tanh')
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    if "hx" in input_dict:
        hx = torch.tensor(input_dict["hx"])
    else:
        hx = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        if hx is not None:
          hx = hx.cuda()

    rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    output, hn = rnn(input_tensor, hx)

    if not cpu:
        output = output.cpu()
        hn = hn.cpu()

    return {"output": output.detach().numpy(), "hn": hn.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    hidden_size = input_dict["hidden_size"]
    input_size = input_dict["input_size"]
    num_layers = input_dict.get("num_layers", 1)
    nonlinearity = input_dict.get("nonlinearity", 'tanh')
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    if "hx" in input_dict:
        hx = tf.constant(input_dict["hx"])
    else:
        if batch_first:
            N = input_tensor.shape[0]
        else:
            N = input_tensor.shape[1]
        hx = tf.zeros((num_layers, N, hidden_size))

    if batch_first:
        input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

    if nonlinearity == 'tanh':
        activation = 'tanh'
    else:
        activation = 'relu'

    rnn_layers = []
    for i in range(num_layers):
        rnn_layers.append(tf.keras.layers.SimpleRNN(hidden_size, activation=activation, use_bias=bias, return_sequences=True, return_state=True))

    output = input_tensor
    states = []
    initial_states = tf.unstack(hx, axis=0)

    for i, rnn_layer in enumerate(rnn_layers):
        output, state = rnn_layer(output, initial_state=initial_states[i])
        states.append(state)
        output = output

    hn = tf.stack(states, axis=0)

    if batch_first:
        output = tf.transpose(output, perm=[1, 0, 2])

    return {"output": output.numpy(), "hn": hn.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "input_size": 10,
        "hidden_size": 20,
        "num_layers": 2,
        "hx": np.random.rand(2, 3, 20).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Output results do not match"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Hidden state results do not match"

    print("Success")

if __name__ == "__main__":
    main()