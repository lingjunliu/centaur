import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    h_0 = torch.tensor(input_dict["h_0"])
    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        h_0 = h_0.cuda()

    rnn = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    output, h_n = rnn(input_tensor, h_0)

    if not cpu:
        output = output.cpu()
        h_n = h_n.cpu()

    return {"output": output.detach().numpy(), "h_n": h_n.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        h_0 = tf.constant(input_dict["h_0"], dtype=tf.float32)
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

        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])
            
        rnn_cells = [tf.keras.layers.GRUCell(hidden_size) for _ in range(num_layers)]
        if num_layers > 1:
            rnn = tf.keras.layers.StackedRNNCells(rnn_cells)
        else:
            rnn = rnn_cells[0]

        gru = tf.keras.layers.RNN(rnn, return_sequences=True, return_state=True)
        
        if num_layers > 1:
            h0_split = tf.split(h_0, num_layers, axis=0)
            h0_list = [tf.squeeze(h, axis=0) for h in h0_split]
            output, *last_states = gru(input_tensor, initial_state=h0_list)
            last_states = tf.stack(last_states, axis=0)
        else:
          
            output, last_states = gru(input_tensor, initial_state=tf.split(tf.squeeze(h_0, axis=0), num_or_size_splits=3, axis=0))
            last_states = tf.expand_dims(tf.stack(last_states, axis = 0), axis = 0)

        if batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])

        output = output.numpy()
        last_states = last_states.numpy()

    return {"output": output, "h_n": last_states}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "h_0": np.random.rand(1, 3, 20).astype(np.float32),
        "input_size": 10,
        "hidden_size": 20,
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Results do not match in output"
    assert np.allclose(torch_result["h_n"], tf_result["h_n"], atol=A_TOL), "Results do not match in h_n"

    print("Success")

if __name__ == "__main__":
    main()