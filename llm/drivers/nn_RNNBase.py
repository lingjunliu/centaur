import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    mode = input_dict["mode"]
    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    proj_size = input_dict.get("proj_size", 0)
    
    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()

    if mode == "RNN_RELU":
        rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, nonlinearity='relu')
    elif mode == "RNN_TANH":
        rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, nonlinearity='tanh')
    elif mode == "LSTM":
        rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    elif mode == "GRU":
        rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not cpu:
        rnn = rnn.cuda()
        
    result, hn = rnn(input_tensor, hx)
    if isinstance(hn, tuple):
        hn = hn[0]

    if not cpu:
        result = result.cpu()
        hn = hn.cpu()
    
    return {"result": result.detach().numpy(), "hn": hn.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        mode = input_dict["mode"]
        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]
        num_layers = input_dict.get("num_layers", 1)
        bias = input_dict.get("bias", True)
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)
        proj_size = input_dict.get("proj_size", 0)

        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hx = tf.constant(input_dict["hx"], dtype=tf.float32)
        
        if mode == "RNN_RELU":
            cell = tf.keras.layers.SimpleRNN(hidden_size, activation="relu", use_bias=bias, dropout=dropout, return_sequences=True, return_state=True)
        elif mode == "RNN_TANH":
            cell = tf.keras.layers.SimpleRNN(hidden_size, activation="tanh", use_bias=bias, dropout=dropout, return_sequences=True, return_state=True)
        elif mode == "LSTM":
            cell = tf.keras.layers.LSTM(hidden_size, use_bias=bias, dropout=dropout, return_sequences=True, return_state=True)
        elif mode == "GRU":
            cell = tf.keras.layers.GRU(hidden_size, use_bias=bias, dropout=dropout, return_sequences=True, return_state=True)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "LSTM":
            num_states = 2
        else:
            num_states = 1
        
        rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True, stateful=False)

        if num_layers > 1 and mode in ("LSTM", "GRU"):
            initial_states = [hx for _ in range(num_layers)]
        else:
            initial_states = [hx]
        
        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])
        
        outputs = rnn(input_tensor, initial_state=initial_states)

        if isinstance(outputs, tuple):
            outputs, last_state = outputs[0], outputs[1:]
        else:
            last_state = outputs[1:]
            outputs = outputs[0]

        if batch_first:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        result = outputs.numpy()

        if num_states > 1:
            hn = np.concatenate([s.numpy() for s in last_state], axis=-1) if isinstance(last_state, (list, tuple)) else last_state.numpy()
        else:
             hn = last_state[0].numpy() if isinstance(last_state, (list, tuple)) else last_state.numpy()

    return {"result": result, "hn": hn}

def main():
    A_TOL = 0.01

    input_data = {
        "mode": "RNN_RELU",
        "input_size": 10,
        "hidden_size": 5,
        "num_layers": 1,
        "input": np.random.randn(2, 3, 10).astype(np.float32), 
        "hx": np.random.randn(1, 3, 5).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match (result)"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Results do not match (hn)"

    print("Success")

if __name__ == "__main__":
    main()