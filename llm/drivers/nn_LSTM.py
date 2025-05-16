import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    proj_size = input_dict.get("proj_size", 0)

    input_tensor = torch.tensor(input_dict["input"])
    if "h_0" in input_dict:
        h_0 = torch.tensor(input_dict["h_0"])
        c_0 = torch.tensor(input_dict["c_0"])
    else:
        h_0 = torch.zeros(num_layers, input_tensor.shape[1], hidden_size)
        c_0 = torch.zeros(num_layers, input_tensor.shape[1], hidden_size)
        h_0 = torch.tensor(h_0)
        c_0 = torch.tensor(c_0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if h_0 is not None:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

    rnn = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size)

    if h_0 is not None:
        output, (h_n, c_n) = rnn(input_tensor, (h_0, c_0))
    else:
        output, (h_n, c_n) = rnn(input_tensor)

    if not cpu:
        output = output.cpu()
        h_n = h_n.cpu()
        c_n = c_n.cpu()

    return {"output": output.detach().numpy(), "h_n": h_n.detach().numpy(), "c_n": c_n.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]
        num_layers = input_dict.get("num_layers", 1)
        bias = input_dict.get("bias", True)
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)
        proj_size = input_dict.get("proj_size", 0)

        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        
        if "h_0" in input_dict:
            h_0 = tf.constant(input_dict["h_0"], dtype=tf.float32)
            c_0 = tf.constant(input_dict["c_0"], dtype=tf.float32)
        else:
            h_0 = tf.zeros((num_layers, input_tensor.shape[1], hidden_size), dtype=tf.float32)
            c_0 = tf.zeros((num_layers, input_tensor.shape[1], hidden_size), dtype=tf.float32)

        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

        lstm_layers = []
        for _ in range(num_layers):
            lstm_layers.append(tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True))

        output = input_tensor
        h_states = []
        c_states = []

        if h_0 is not None:
          initial_states = [(h_0[i, :input_tensor.shape[1], :], c_0[i, :input_tensor.shape[1], :]) for i in range(num_layers)]
        else:
          initial_states = None

        for i, layer in enumerate(lstm_layers):
            if initial_states is not None:
                output, h, c = layer(output, initial_state=initial_states[i])
            else:
                output, h, c = layer(output)
            h_states.append(h)
            c_states.append(c)

            if dropout > 0:
                output = tf.keras.layers.Dropout(dropout)(output)

        h_n = tf.stack(h_states, axis=0)
        c_n = tf.stack(c_states, axis=0)

        if batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])

        output = output.numpy()
        h_n = h_n.numpy()
        c_n = c_n.numpy()

    return {"output": output, "h_n": h_n, "c_n": c_n}


def main():
    A_TOL = 0.01

    input_data = {
        "input_size": 10,
        "hidden_size": 20,
        "num_layers": 1,
        "input": np.random.randn(5, 3, 10).astype(np.float32),
        "h_0": np.random.randn(1, 3, 20).astype(np.float32),
        "c_0": np.random.randn(1, 3, 20).astype(np.float32),
        "batch_first": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Output results do not match"
    assert np.allclose(torch_result["h_n"], tf_result["h_n"], atol=A_TOL), "h_n results do not match"
    assert np.allclose(torch_result["c_n"], tf_result["c_n"], atol=A_TOL), "c_n results do not match"

    print("Success")

if __name__ == "__main__":
    main()