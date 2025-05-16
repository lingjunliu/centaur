import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["h0"])
    input_size = input_dict.get("input_size")
    hidden_size = input_dict.get("hidden_size")
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()

    gru = torch.nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    with torch.no_grad():
        output, hn = gru(input_tensor, hidden_state)

    if not cpu:
        output = output.cpu()
        hn = hn.cpu()

    return {"output": output.numpy(), "hn": hn.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    hidden_state = tf.constant(input_dict["h0"])
    input_size = input_dict.get("input_size")
    hidden_size = input_dict.get("hidden_size")
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)

    if batch_first:
        input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

    units = hidden_size

    if bidirectional:
        units = hidden_size

    gru = tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               activation='tanh',
                               recurrent_activation='sigmoid',
                               use_bias=bias,
                               dropout=dropout)
    if bidirectional:
        gru = tf.keras.layers.Bidirectional(gru)
        
    if num_layers > 1:
        initial_state = tf.split(hidden_state, num_layers, axis=0)
        initial_state = [tf.reshape(s, (hidden_state.shape[1], hidden_size)) for s in initial_state]
    else:
        initial_state = tf.reshape(hidden_state, (hidden_state.shape[1], hidden_size))

    if not bidirectional:
        output, hn = gru(input_tensor, initial_state=initial_state)
    else:
        output, hf, hb = gru(input_tensor, initial_state=initial_state)
        hn = tf.concat([hf, hb], axis=-1)

    if batch_first:
        output = tf.transpose(output, perm=[1, 0, 2])

    output = output.numpy()
    if num_layers > 1:
        hn = tf.stack(initial_state, axis=0).numpy()
    else:
        hn = hn.numpy()
    
    hn = np.expand_dims(hn, axis=0)

    return {"output": output, "hn": hn}

def main():
    A_TOL = 0.01

    input_data = {
        "input_size": 10,
        "hidden_size": 20,
        "input": np.random.randn(5, 3, 10).astype(np.float32),
        "h0": np.random.randn(1, 3, 20).astype(np.float32),
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Results do not match (output)"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Results do not match (hn)"

    input_data = {
        "input_size": 10,
        "hidden_size": 20,
        "input": np.random.randn(3, 5, 10).astype(np.float32),
        "h0": np.random.randn(1, 3, 20).astype(np.float32),
        "num_layers": 1,
        "bias": True,
        "batch_first": True,
        "dropout": 0.0,
        "bidirectional": False,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Results do not match (output)"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Results do not match (hn)"
    
    input_data = {
        "input_size": 10,
        "hidden_size": 20,
        "input": np.random.randn(5, 3, 10).astype(np.float32),
        "h0": np.random.randn(2, 3, 20).astype(np.float32),
        "num_layers": 2,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Results do not match (output)"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Results do not match (hn)"
    
    input_data = {
        "input_size": 10,
        "hidden_size": 20,
        "input": np.random.randn(5, 3, 10).astype(np.float32),
        "h0": np.random.randn(1, 3, 20).astype(np.float32),
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": True,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Results do not match (output)"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Results do not match (hn)"

    print("Success")

if __name__ == "__main__":
    main()