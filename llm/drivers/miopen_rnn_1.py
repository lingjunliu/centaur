import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = [torch.tensor(w) for w in input_dict["weight"]]
    bias_tensor = torch.tensor(input_dict["bias"])
    hx_tensor = torch.tensor(input_dict["hx"])
    cx_tensor = torch.tensor(input_dict["cx"])

    mode_str = input_dict.get("mode", "LSTM")
    if mode_str == "LSTM":
        mode = 0
    elif mode_str == "GRU":
        mode = 1
    elif mode_str == "RNN_TANH":
        mode = 2
    elif mode_str == "RNN_RELU":
        mode = 3
    else:
        raise ValueError("Invalid mode string.")
    input_size = input_dict.get("input_size")
    hidden_size = input_dict.get("hidden_size")
    num_layers = input_dict.get("num_layers", 1)
    dropout = input_dict.get("dropout", 0.0)
    train = input_dict.get("train", True)
    bidirectional = input_dict.get("bidirectional", False)
    batch_first = input_dict.get("batch_first", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = [w.cuda() for w in weight_tensor]
        bias_tensor = bias_tensor.cuda()
        hx_tensor = hx_tensor.cuda()
        cx_tensor = cx_tensor.cuda()

    result = torch.miopen_rnn(input_tensor, weight_tensor, num_layers, 
                               hx_tensor, cx_tensor, mode, input_size, 
                               hidden_size, bias_tensor, dropout, train, 
                               bidirectional, bool(batch_first))

    if not cpu:
        result = (result[0].cpu(), result[1].cpu(), result[2].cpu())

    return {"result0": result[0].numpy(), "result1": result[1].numpy(), "result2": result[2].numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight_tensor = [tf.constant(w, dtype=tf.float32) for w in input_dict["weight"]]
        bias_tensor = tf.constant(input_dict["bias"], dtype=tf.float32)
        hx_tensor = tf.constant(input_dict["hx"], dtype=tf.float32)
        cx_tensor = tf.constant(input_dict["cx"], dtype=tf.float32)

        mode_str = input_dict.get("mode", "LSTM")
        input_size = input_dict.get("input_size")
        hidden_size = input_dict.get("hidden_size")
        num_layers = input_dict.get("num_layers", 1)
        dropout = input_dict.get("dropout", 0.0)
        train = input_dict.get("train", True)
        bidirectional = input_dict.get("bidirectional", False)
        batch_first = input_dict.get("batch_first", False)

        if mode_str == "LSTM":
            rnn_layer = tf.keras.layers.LSTM(hidden_size, 
                                               return_sequences=True, 
                                               return_state=True,
                                               stateful=False,
                                               batch_input_shape=(input_tensor.shape[0] if batch_first else input_tensor.shape[1] , input_tensor.shape[1] if batch_first else input_tensor.shape[0] ,input_size))

            rnn_layer.build(input_shape=(input_tensor.shape[0] if batch_first else input_tensor.shape[1] , input_tensor.shape[1] if batch_first else input_tensor.shape[0] ,input_size))

            lstm_weights = []
            lstm_weights.append(np.transpose(weight_tensor[0].numpy().reshape((4, hidden_size, input_size)), axes=(0, 2, 1)).reshape((4*hidden_size, input_size)))
            lstm_weights.append(np.transpose(weight_tensor[1].numpy().reshape((4, hidden_size, hidden_size)), axes=(0, 2, 1)).reshape((4*hidden_size, hidden_size)))
            lstm_weights.append(bias_tensor.numpy())

            rnn_layer.set_weights(lstm_weights)


            if bool(batch_first):
              input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

            outputs, h_state, c_state = rnn_layer(input_tensor, initial_state=[hx_tensor, cx_tensor])

            if bool(batch_first):
              outputs = tf.transpose(outputs, perm=[1, 0, 2])

            result0 = outputs.numpy()
            result1 = h_state.numpy()
            result2 = c_state.numpy()
        else:
            raise ValueError("Only LSTM mode is currently supported in TensorFlow version.")

    return {"result0": result0, "result1": result1, "result2": result2}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(5, 3, 10).astype(np.float32),
        "weight": [np.random.randn(4*20, 10).astype(np.float32), np.random.randn(4*20,20).astype(np.float32)],
        "bias": np.random.randn(4*20).astype(np.float32),
        "hx": np.random.randn(3, 20).astype(np.float32),
        "cx": np.random.randn(3, 20).astype(np.float32),
        "mode": "LSTM",
        "input_size": 10,
        "hidden_size": 20,
        "num_layers": 1,
        "dropout": 0.0,
        "train": False,
        "bidirectional": False,
        "batch_first": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result0"], tf_result["result0"], atol=A_TOL), "Results0 do not match"
    assert np.allclose(torch_result["result1"], tf_result["result1"], atol=A_TOL), "Results1 do not match"
    assert np.allclose(torch_result["result2"], tf_result["result2"], atol=A_TOL), "Results2 do not match"

    print("Success")

if __name__ == "__main__":
    main()