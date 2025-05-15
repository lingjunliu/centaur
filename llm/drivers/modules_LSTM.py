import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state_h = torch.tensor(input_dict["hx"][0])
    hidden_state_c = torch.tensor(input_dict["hx"][1])
    hidden_state = (hidden_state_h, hidden_state_c)
    
    input_size = input_dict.get("input_size")
    hidden_size = input_dict.get("hidden_size")
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state_h = hidden_state_h.cuda()
        hidden_state_c = hidden_state_c.cuda()
        hidden_state = (hidden_state_h, hidden_state_c)
        lstm = lstm.cuda()

    result, hn = lstm(input_tensor, hidden_state)
    
    if not cpu:
        result = result.cpu()
        hn = (hn[0].cpu(), hn[1].cpu())
    
    return {"result": result.detach().numpy(), "hn": (hn[0].detach().numpy(), hn[1].detach().numpy())}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    hidden_state_h = tf.constant(input_dict["hx"][0])
    hidden_state_c = tf.constant(input_dict["hx"][1])
    
    input_size = input_dict.get("input_size")
    hidden_size = input_dict.get("hidden_size")
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    seq_len = input_tensor.shape[0] if not batch_first else input_tensor.shape[1]
    batch_size = input_tensor.shape[1] if not batch_first else input_tensor.shape[0]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        lstm_cells = []
        for _ in range(num_layers):
            lstm_cells.append(tf.keras.layers.LSTMCell(hidden_size, use_bias=bias, dropout=dropout))

        if bidirectional:
            lstm_cells_bw = []
            for _ in range(num_layers):
                lstm_cells_bw.append(tf.keras.layers.LSTMCell(hidden_size, use_bias=bias, dropout=dropout))

            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.StackedRNNCells(lstm_cells), backward_layer=tf.keras.layers.StackedRNNCells(lstm_cells_bw), merge_mode='concat')

        else:
            lstm = tf.keras.layers.StackedRNNCells(lstm_cells)

        if num_layers > 1:
            initial_state = [hidden_state_h, hidden_state_c]
        else:
            initial_state = [tf.reshape(hidden_state_h, (batch_size, hidden_size)), tf.reshape(hidden_state_c, (batch_size, hidden_size))]
        
        lstm_layer = tf.keras.layers.RNN(lstm, return_sequences=True, return_state=True)

        if batch_first:
            result, final_h, final_c = lstm_layer(input_tensor, initial_state=initial_state)
        else:
            result, final_h, final_c = lstm_layer(tf.transpose(input_tensor, perm=[1, 0, 2]), initial_state=initial_state)
            result = tf.transpose(result, perm=[1, 0, 2])

        result = result.numpy()
        hn_h = np.stack([final_h], axis=0)
        hn_c = np.stack([final_c], axis=0)
        
    return {"result": result, "hn": (hn_h, hn_c)}

def main():
    A_TOL = 0.1
    seq_len = 5
    batch_size = 2
    input_size = 10
    hidden_size = 20
    num_layers = 1
    
    input_data = {
        "input": np.random.rand(seq_len, batch_size, input_size).astype(np.float32),
        "hx": (np.random.rand(num_layers, batch_size, hidden_size).astype(np.float32), np.random.rand(num_layers, batch_size, hidden_size).astype(np.float32)),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "batch_first": False,
        "bidirectional": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["hn"][0], tf_result["hn"][0], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["hn"][1], tf_result["hn"][1], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()