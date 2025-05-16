import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hidden = torch.tensor(input_dict["h0"]) if "h0" in input_dict else None
    seq_length = input_dict.get("seq_length", input_tensor.size(0))
    batch_size = input_dict.get("batch_size", input_tensor.size(1) if input_tensor.ndim > 1 else 1)
    input_size = input_dict.get("input_size", input_tensor.size(2) if input_tensor.ndim > 2 else input_tensor.size(1) if input_tensor.ndim > 1 else 1)
    hidden_size = input_dict.get("hidden_size", 1)
    num_layers = input_dict.get("num_layers", 1)
    bias = input_dict.get("bias", True)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if hidden is not None:
            hidden = hidden.cuda()

    gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)
    
    if hidden is not None:
        result, hn = gru(input_tensor, hidden)
    else:
        result, hn = gru(input_tensor)

    if not cpu:
        result = result.cpu()
        if hn is not None:
            hn = hn.cpu()

    return {"result": result.detach().numpy(), "hn": hn.detach().numpy() if hn is not None else None}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hidden = tf.constant(input_dict["h0"], dtype=tf.float32) if "h0" in input_dict else None
        seq_length = input_dict.get("seq_length", input_tensor.shape[0])
        batch_size = input_dict.get("batch_size", input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1)
        input_size = input_dict.get("input_size", input_tensor.shape[2] if len(input_tensor.shape) > 2 else input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1)
        hidden_size = input_dict.get("hidden_size", 1)
        num_layers = input_dict.get("num_layers", 1)
        bias = input_dict.get("bias", True)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)
        
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        
        if hidden is not None:
            hidden = tf.transpose(hidden, perm=[1, 0, 2])
            initial_state = [hidden[i] for i in range(num_layers)]
        else:
            initial_state = None

        gru = tf.keras.layers.GRU(hidden_size,
                                   return_sequences=True,
                                   return_state=True,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   use_bias=bias,
                                   dropout=dropout,
                                   unroll=False,
                                   reset_after=True,
                                   go_backwards=False,
                                   stateful=False,
                                   kernel_initializer='glorot_uniform',
                                   recurrent_initializer='orthogonal',
                                   bias_initializer='zeros',
                                   kernel_regularizer=None,
                                   recurrent_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   kernel_constraint=None,
                                   recurrent_constraint=None,
                                   bias_constraint=None)

        
        if num_layers > 1:
            gru_layers = [tf.keras.layers.GRU(hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       activation='tanh',
                                       recurrent_activation='sigmoid',
                                       use_bias=bias,
                                       dropout=dropout,
                                       unroll=False,
                                       reset_after=True,
                                       go_backwards=False,
                                       stateful=False,
                                       kernel_initializer='glorot_uniform',
                                       recurrent_initializer='orthogonal',
                                       bias_initializer='zeros',
                                       kernel_regularizer=None,
                                       recurrent_regularizer=None,
                                       bias_regularizer=None,
                                       activity_regularizer=None,
                                       kernel_constraint=None,
                                       recurrent_constraint=None,
                                       bias_constraint=None) for _ in range(num_layers)]
        
            outputs = input_tensor
            states = []

            for i in range(num_layers):
                output, state = gru_layers[i](outputs, initial_state=initial_state[i] if initial_state else None)
                outputs = output
                states.append(state)

            result = outputs
            hn = tf.stack(states, axis=1)
            hn = tf.transpose(hn, perm=[1, 0, 2])
        else:
            if initial_state is not None:
                result, hn = gru(input_tensor, initial_state=initial_state)
            else:
                result, hn = gru(input_tensor)
            if hidden is not None:
                hn = tf.transpose(hn, perm=[1, 0, 2])
        
        result = result.numpy()
        hn = hn.numpy() if hidden is not None else None
        
    return {"result": result, "hn": hn}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
        "h0": np.array([[[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32),
        "hidden_size": 2,
        "num_layers": 2
    }

    torch_result = torch_version(input_data)
    
    input_data_tf = {k:v for k, v in input_data.items()}

    input_data_tf["h0"] = np.transpose(input_data["h0"], axes=[1,0,2])
    tf_result = tensorflow_version(input_data_tf)
    
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()