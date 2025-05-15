import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch import nn

    input_tensor = torch.tensor(input_dict["input"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict.get("bias_ih", None)) if input_dict.get("bias_ih", None) is not None else None
    bias_hh = torch.tensor(input_dict.get("bias_hh", None)) if input_dict.get("bias_hh", None) is not None else None
    init_h = torch.tensor(input_dict["init_h"])
    init_c = torch.tensor(input_dict.get("init_c", None)) if input_dict.get("init_c", None) is not None else None
    mode = input_dict.get("mode", 'LSTM')
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    train = input_dict.get("train", True)
    bidirectional = input_dict.get("bidirectional", False)
    num_layers = input_dict.get("num_layers", 1)
    
    num_directions = 2 if bidirectional else 1
    hidden_size = weight_hh.shape[1] // 4 if mode == 'LSTM' else weight_hh.shape[1] // 3 if mode == 'GRU' else weight_hh.shape[1]

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        if bias_ih is not None:
            bias_ih = bias_ih.cuda()
        if bias_hh is not None:
            bias_hh = bias_hh.cuda()
        init_h = init_h.cuda()
        if init_c is not None:
            init_c = init_c.cuda()

    if mode == 'LSTM':
        rnn = nn.LSTM(input_size=input_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers,
                      batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    elif mode == 'GRU':
        rnn = nn.GRU(input_size=input_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers,
                     batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    elif mode == 'RNN_TANH':
        rnn = nn.RNN(input_size=input_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers,
                     batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, nonlinearity='tanh')
    elif mode == 'RNN_RELU':
        rnn = nn.RNN(input_size=input_tensor.shape[2], hidden_size=hidden_size, num_layers=num_layers,
                     batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, nonlinearity='relu')
    else:
        raise ValueError("Unsupported RNN mode: {}".format(mode))

    
    #rnn.weight_ih_l0 = nn.Parameter(weight_ih)
    #rnn.weight_hh_l0 = nn.Parameter(weight_hh)
    #if bias_ih is not None:
    #    rnn.bias_ih_l0 = nn.Parameter(bias_ih)
    #if bias_hh is not None:
    #    rnn.bias_hh_l0 = nn.Parameter(bias_hh)
        
    if not cpu:
        rnn = rnn.cuda()
    
    init_h = init_h.reshape(num_layers * num_directions, batch_size, hidden_size)
    if init_c is not None:
        init_c = init_c.reshape(num_layers * num_directions, batch_size, hidden_size)

    output, (h_n, c_n) = rnn(input_tensor, (init_h, init_c) if init_c is not None else (init_h,))

    if not cpu:
        output = output.cpu()
        h_n = h_n.cpu()
        if c_n is not None:
            c_n = c_n.cpu()
    
    return {"result": (output.detach().numpy(), h_n.detach().numpy(), c_n.detach().numpy() if init_c is not None else None)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = tf.constant(input_dict.get("bias_ih", np.zeros_like(input_dict["weight_ih"][:, 0]))) if input_dict.get("bias_ih", None) is not None else None
        bias_hh = tf.constant(input_dict.get("bias_hh", np.zeros_like(input_dict["weight_hh"][:, 0]))) if input_dict.get("bias_hh", None) is not None else None
        init_h = tf.constant(input_dict["init_h"])
        init_c = tf.constant(input_dict.get("init_c", np.zeros_like(input_dict["init_h"]))) if input_dict.get("init_c", None) is not None else None
        mode = input_dict.get("mode", 'LSTM')
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        train = input_dict.get("train", True)
        bidirectional = input_dict.get("bidirectional", False)
        num_layers = input_dict.get("num_layers", 1)
        
        num_directions = 2 if bidirectional else 1
        hidden_size = weight_hh.shape[1] // 4 if mode == 'LSTM' else weight_hh.shape[1] // 3 if mode == 'GRU' else weight_hh.shape[1]

        lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        
        if batch_first:
          input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])
          
        outputs, h, c = lstm(input_tensor, initial_state=[tf.reshape(init_h, (num_layers * num_directions, batch_size, hidden_size)), tf.reshape(init_c, (num_layers * num_directions, batch_size, hidden_size)) if init_c is not None else tf.zeros((num_layers * num_directions, batch_size, hidden_size))])
          
        if batch_first:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
            
        h_n = h
        c_n = c if init_c is not None else None

        return {"result": (outputs.numpy(), h_n.numpy(), c_n.numpy() if init_c is not None else None)}

def main():
    A_TOL = 0.01
    input_size = 10
    hidden_size = 20
    batch_size = 5
    seq_len = 3

    input_data = {
        "input": np.random.randn(seq_len, batch_size, input_size).astype(np.float32),
        "weight_ih": np.random.randn(4 * hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.randn(4 * hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.randn(4 * hidden_size).astype(np.float32),
        "bias_hh": np.random.randn(4 * hidden_size).astype(np.float32),
        "init_h": np.random.randn(1, batch_size, hidden_size).astype(np.float32),
        "init_c": np.random.randn(1, batch_size, hidden_size).astype(np.float32),
        "mode": "LSTM",
        "batch_first": False,
        "dropout": 0.0,
        "train": True,
        "bidirectional": False,
        "num_layers": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"][0], tf_result["result"][0], atol=A_TOL), "Output mismatch"
    assert np.allclose(torch_result["result"][1], tf_result["result"][1], atol=A_TOL), "h_n mismatch"
    if torch_result["result"][2] is not None:
        assert np.allclose(torch_result["result"][2], tf_result["result"][2], atol=A_TOL), "c_n mismatch"

    print("Success")

if __name__ == "__main__":
    main()