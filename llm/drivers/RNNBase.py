import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_size = input_dict.get("hidden_size")
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        
    rnn = torch.nn.RNN(
        input_size=input_tensor.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional
    )

    for name, param in rnn.named_parameters():
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)

    h0 = torch.zeros(num_layers * (2 if bidirectional else 1), input_tensor.size(0 if batch_first else 1), hidden_size)
    if not cpu:
      h0 = h0.cuda()

    result, hn = rnn(input_tensor, h0)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hidden_size = input_dict.get("hidden_size")
        num_layers = input_dict.get("num_layers", 1)
        bias = input_dict.get("bias", True)
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)

        if batch_first:
          pass
        else:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

        rnn_cells = [tf.keras.layers.SimpleRNNCell(hidden_size, activation='tanh') for _ in range(num_layers)]
        if dropout > 0.0:
            rnn_cells = [tf.keras.layers.Dropout(dropout)(cell) for cell in rnn_cells]
        
        if num_layers > 1:
            rnn = tf.keras.layers.StackedRNNCells(rnn_cells)
        else:
            rnn = rnn_cells[0]
        
        layer = tf.keras.layers.RNN(rnn, return_sequences=True, return_state=True)

        initial_state = [tf.zeros(shape=(input_tensor.shape[1], hidden_size), dtype=tf.float32) for _ in range(num_layers)]

        output, *state = layer(input_tensor, initial_state=initial_state)

        if not batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])

        result = output.numpy()
        hn = np.stack(state, axis=0).numpy()

    return {"result": result, "hn": hn}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "hidden_size": 7,
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Hidden states do not match"

    print("Success")

if __name__ == "__main__":
    main()