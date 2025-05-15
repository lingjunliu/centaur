import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    h_0 = torch.tensor(input_dict["h_0"])
    c_0 = torch.tensor(input_dict["c_0"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", True)
    num_layers = input_dict.get("num_layers", 1)
    dropout = input_dict.get("dropout", 0.0)
    train = input_dict.get("train", True)
    bidirectional = input_dict.get("bidirectional", False)
    batch_first = input_dict.get("batch_first", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        h_0 = h_0.cuda()
        c_0 = c_0.cuda()
        weight = weight.cuda()

    lstm = torch.nn.LSTM(input_size=input_tensor.shape[-1], hidden_size=h_0.shape[-1], num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)
    
    lstm.weight_ih_l0 = torch.nn.Parameter(weight[:lstm.hidden_size * 4, :input_tensor.shape[-1]])
    lstm.weight_hh_l0 = torch.nn.Parameter(weight[lstm.hidden_size * 4:lstm.hidden_size * 8, :lstm.hidden_size])
    if bias:
        lstm.bias_ih_l0 = torch.nn.Parameter(weight[lstm.hidden_size * 8:lstm.hidden_size * 8 + lstm.hidden_size * 4, -1].reshape(-1))
        lstm.bias_hh_l0 = torch.nn.Parameter(weight[lstm.hidden_size * 8 + lstm.hidden_size * 4:lstm.hidden_size * 12, -1].reshape(-1))
    else:
        lstm.bias_ih_l0 = None
        lstm.bias_hh_l0 = None

    output, (h_n, c_n) = lstm(input_tensor, (h_0, c_0))
    
    if not cpu:
        output = output.cpu()
        h_n = h_n.cpu()
        c_n = c_n.cpu()

    return {"output": output.detach().numpy(), "h_n": h_n.detach().numpy(), "c_n": c_n.detach().numpy()}

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
        weight = tf.constant(input_dict["weight"])
        bias = input_dict.get("bias", True)
        num_layers = input_dict.get("num_layers", 1)
        dropout = input_dict.get("dropout", 0.0)
        train = input_dict.get("train", True)
        bidirectional = input_dict.get("bidirectional", False)
        batch_first = input_dict.get("batch_first", False)
    
        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])
        
        hidden_size = h_0.shape[-1]

        w_i = weight[:hidden_size * 4, :input_tensor.shape[-1]]
        w_h = weight[hidden_size * 4:hidden_size * 8, :hidden_size]
        b_i = weight[hidden_size * 8:hidden_size * 8 + hidden_size * 4, -1]
        b_h = weight[hidden_size * 8 + hidden_size * 4:, -1]
        
        def lstm_cell(input, hidden_state, cell_state):
            linear = tf.matmul(input, w_i) + tf.matmul(hidden_state, w_h) + b_i + b_h
            i, f, g, o = tf.split(linear, num_or_size_splits=4, axis=1)
            
            i = tf.sigmoid(i)
            f = tf.sigmoid(f)
            g = tf.tanh(g)
            o = tf.sigmoid(o)
            
            new_cell_state = f * cell_state + i * g
            new_hidden_state = o * tf.tanh(new_cell_state)
            
            return new_hidden_state, new_cell_state
            
        output_list = []
        hidden_state = h_0[0]
        cell_state = c_0[0]

        for i in range(input_tensor.shape[0]):
            hidden_state, cell_state = lstm_cell(input_tensor[i], hidden_state, cell_state)
            output_list.append(hidden_state)
        
        output = tf.stack(output_list)
        
        if batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])

        h_n = tf.reshape(hidden_state, h_0.shape)
        c_n = tf.reshape(cell_state, c_0.shape)
        
        output = output.numpy()
        h_n = h_n.numpy()
        c_n = c_n.numpy()

        return {"output": output, "h_n": h_n, "c_n": c_n}

def main():
    A_TOL = 0.01
    input_size = 10
    hidden_size = 20
    batch_size = 3
    seq_len = 5

    input_data = {
        "input": np.random.randn(seq_len, batch_size, input_size).astype(np.float32),
        "h_0": np.random.randn(1, batch_size, hidden_size).astype(np.float32),
        "c_0": np.random.randn(1, batch_size, hidden_size).astype(np.float32),
        "weight": np.random.randn(hidden_size * 12, max(input_size, hidden_size)+1).astype(np.float32)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Output results do not match"
    assert np.allclose(torch_result["h_n"], tf_result["h_n"], atol=A_TOL), "h_n results do not match"
    assert np.allclose(torch_result["c_n"], tf_result["c_n"], atol=A_TOL), "c_n results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()