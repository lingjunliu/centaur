import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    nonlinearity = input_dict.get("nonlinearity", 'tanh')
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    if "h0" in input_dict:
        h0 = torch.tensor(input_dict["h0"])
    else:
        h0 = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        if h0 is not None:
            h0 = h0.cuda()
    
    rnn = torch.nn.RNN(input_size=input_tensor.shape[-1], 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       nonlinearity=nonlinearity, 
                       bias=bias, 
                       batch_first=batch_first,
                       dropout=dropout, 
                       bidirectional=bidirectional)
    
    if not cpu:
        rnn = rnn.cuda()

    result, hn = rnn(input_tensor, h0)

    if not cpu:
        result = result.cpu()
        if hn is not None:
            hn = hn.cpu()
    
    if hn is not None:
        return {"result": result.detach().numpy(), "hn": hn.detach().numpy()}
    else:
        return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hidden_size = input_dict["hidden_size"]
        num_layers = input_dict.get("num_layers", 1)
        nonlinearity = input_dict.get("nonlinearity", 'tanh')
        bias = input_dict.get("bias", True)
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)
        
        if "h0" in input_dict:
            h0 = tf.constant(input_dict["h0"], dtype=tf.float32)
        else:
            if batch_first:
                batch_size = input_tensor.shape[0]
            else:
                batch_size = input_tensor.shape[1]
            
            if bidirectional:
                direction = 2
            else:
                direction = 1

            h0 = tf.zeros([num_layers * direction, batch_size, hidden_size], dtype=tf.float32)


        if nonlinearity == 'tanh':
            activation = 'tanh'
        elif nonlinearity == 'relu':
            activation = 'relu'
        else:
            raise ValueError("Unsupported nonlinearity: {}".format(nonlinearity))

        if bidirectional:
            rnn_cell_fw = tf.keras.layers.SimpleRNN(hidden_size, activation=activation, use_bias=bias, dropout=dropout, return_sequences=True, return_state=False)
            rnn_cell_bw = tf.keras.layers.SimpleRNN(hidden_size, activation=activation, use_bias=bias, dropout=dropout, return_sequences=True, return_state=False)

            rnn = tf.keras.layers.Bidirectional(rnn_cell_fw, backward_layer=rnn_cell_bw)
            
            if batch_first:
              input_shaped = input_tensor
            else:
              input_shaped = tf.transpose(input_tensor, perm=[1, 0, 2])
            
            output = rnn(input_shaped)

            if not batch_first:
              output = tf.transpose(output, perm=[1, 0, 2])

            hn = tf.zeros([2, input_tensor.shape[0 if batch_first else 1], hidden_size], dtype=tf.float32)
            return {"result": output.numpy(), "hn": hn.numpy()}


        else:
            rnn_cells = [tf.keras.layers.SimpleRNNCell(hidden_size, activation=activation, use_bias=bias, dropout=dropout) for _ in range(num_layers)]
            stacked_rnn = tf.keras.layers.StackedRNNCells(rnn_cells)
            rnn = tf.keras.layers.RNN(stacked_rnn, return_sequences=True, return_state=True)

            if batch_first:
              input_shaped = input_tensor
            else:
              input_shaped = tf.transpose(input_tensor, perm=[1, 0, 2])

            if "h0" in input_dict:
                output, hn, _ = rnn(input_shaped, initial_state=tuple(tf.unstack(h0, axis=0)))
            else:
                output, hn = rnn(input_shaped)

            
            if not batch_first:
              output = tf.transpose(output, perm=[1, 0, 2])
            
            hn = tf.stack(hn, axis=0)

        if "h0" in input_dict:
            return {"result": output.numpy(), "hn": hn.numpy()}
        else:
            return {"result": output.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "hidden_size": 8,
        "num_layers": 2,
        "batch_first": True,
        "bidirectional": False
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    if input_data["batch_first"]:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        assert np.allclose(torch_result["result"], np.transpose(tf_result["result"], (1,0,2)), atol=A_TOL), "Results do not match"

    input_data_h0 = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "hidden_size": 8,
        "num_layers": 2,
        "batch_first": True,
        "bidirectional": False,
        "h0": np.random.rand(2, 5, 8).astype(np.float32)
    }
    
    torch_result_h0 = torch_version(input_data_h0)
    tf_result_h0 = tensorflow_version(input_data_h0)
    
    if input_data_h0["batch_first"]:
        assert np.allclose(torch_result_h0["result"], tf_result_h0["result"], atol=A_TOL), "Results do not match"
        assert np.allclose(torch_result_h0["hn"], tf_result_h0["hn"], atol=A_TOL), "Results do not match"
    else:
        assert np.allclose(torch_result_h0["result"], np.transpose(tf_result_h0["result"], (1,0,2)), atol=A_TOL), "Results do not match"
        assert np.allclose(torch_result_h0["hn"], tf_result_h0["hn"], atol=A_TOL), "Results do not match"
    

    input_data_bidirectional = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "hidden_size": 8,
        "num_layers": 1,
        "batch_first": True,
        "bidirectional": True
    }
    
    torch_result_bidirectional = torch_version(input_data_bidirectional)
    tf_result_bidirectional = tensorflow_version(input_data_bidirectional)
    
    if input_data_bidirectional["batch_first"]:
        assert np.allclose(torch_result_bidirectional["result"], tf_result_bidirectional["result"], atol=A_TOL), "Results do not match"
    else:
        assert np.allclose(torch_result_bidirectional["result"], np.transpose(tf_result_bidirectional["result"], (1,0,2)), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()