import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    h_0 = torch.tensor(input_dict["h_0"])
    c_0 = torch.tensor(input_dict["c_0"])
    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        h_0 = h_0.cuda()
        c_0 = c_0.cuda()
        lstm = lstm.cuda()
    
    result, (hn, cn) = lstm(input_tensor, (h_0, c_0))
    
    if not cpu:
        result = result.cpu()
        hn = hn.cpu()
        cn = cn.cpu()
    
    return {"result": result.detach().numpy(), "hn": hn.detach().numpy(), "cn": cn.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        h_0 = tf.constant(input_dict["h_0"])
        c_0 = tf.constant(input_dict["c_0"])
        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]
        num_layers = input_dict.get("num_layers", 1)
        bias = input_dict.get("bias", True)
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        
        lstm_layers = []
        for i in range(num_layers):
            lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True,
                                          dropout=dropout,
                                          recurrent_dropout=0.0,
                                          unroll=False,
                                          use_bias=bias,
                                          go_backwards=False,
                                          stateful=False,
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal',
                                          bias_initializer='zeros',
                                          unit_forget_bias=True,
                                          activation='tanh',
                                          recurrent_activation='sigmoid')
            lstm_layers.append(lstm)

        if batch_first:
            inputs = tf.transpose(input_tensor, perm=[1, 0, 2])
        else:
            inputs = input_tensor

        hn_list = []
        cn_list = []
        output = inputs
        
        h_state = tf.unstack(h_0, axis=0)
        c_state = tf.unstack(c_0, axis=0)
        
        for layer in range(num_layers):
            lstm = lstm_layers[layer]
            output, h_state, c_state = lstm(output, initial_state=[h_state[0], c_state[0]])
            hn_list.append(h_state)
            cn_list.append(c_state)

        hn = tf.stack(hn_list)
        cn = tf.stack(cn_list)

        if batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])
        
        output = output.numpy()
        hn = hn.numpy()
        cn = cn.numpy()
    
    return {"result": output, "hn": hn, "cn": cn}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "h_0": np.random.rand(1, 3, 5).astype(np.float32),
        "c_0": np.random.rand(1, 3, 5).astype(np.float32),
        "input_size": 10,
        "hidden_size": 5,
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Hidden state results do not match"
    assert np.allclose(torch_result["cn"], tf_result["cn"], atol=A_TOL), "Cell state results do not match"

    print("Success")

if __name__ == "__main__":
    main()