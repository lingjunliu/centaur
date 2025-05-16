import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["h0"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict.get("bias_ih", torch.zeros(3 * input_dict["hidden_size"])))
    bias_hh = torch.tensor(input_dict.get("bias_hh", torch.zeros(3 * input_dict["hidden_size"])))
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    num_layers = input_dict.get("num_layers", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        bias_ih = bias_ih.cuda()
        bias_hh = bias_hh.cuda()
    
    if num_layers > 1 or bidirectional:
      raise ValueError("Only single-layer unidirectional GRU is supported for dynamic quantization in this example")

    gru = torch.nn.GRU(input_size=input_dict["input_size"], hidden_size=input_dict["hidden_size"], num_layers=num_layers,
                       batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)

    gru.weight_ih_l0 = torch.nn.Parameter(weight_ih)
    gru.weight_hh_l0 = torch.nn.Parameter(weight_hh)
    gru.bias_ih_l0 = torch.nn.Parameter(bias_ih)
    gru.bias_hh_l0 = torch.nn.Parameter(bias_hh)
    
    gru.eval()

    with torch.no_grad():
        output, hn = gru(input_tensor, hidden_state)

    if not cpu:
        output = output.cpu()
        hn = hn.cpu()
    
    return {"output": output.numpy(), "hn": hn.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hidden_state = tf.constant(input_dict["h0"])
        weight_ih = tf.constant(input_dict["weight_ih"].T)
        weight_hh = tf.constant(input_dict["weight_hh"].T)
        bias_ih = tf.constant(input_dict.get("bias_ih", np.zeros(3 * input_dict["hidden_size"])))
        bias_hh = tf.constant(input_dict.get("bias_hh", np.zeros(3 * input_dict["hidden_size"])))
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)
        num_layers = input_dict.get("num_layers", 1)
        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]

        if num_layers > 1 or bidirectional:
            raise ValueError("Only single-layer unidirectional GRU is supported for dynamic quantization in this example")
        
        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])
            
        def gru_cell(input_t, state_t, weight_ih, weight_hh, bias_ih, bias_hh):
            w_ir = weight_ih[:hidden_size, :]
            w_iz = weight_ih[hidden_size:2*hidden_size, :]
            w_in = weight_ih[2*hidden_size:, :]
            
            w_hr = weight_hh[:hidden_size, :]
            w_hz = weight_hh[hidden_size:2*hidden_size, :]
            w_hn = weight_hh[2*hidden_size:, :]

            b_ir = tf.reshape(bias_ih[:hidden_size], (hidden_size,))
            b_iz = tf.reshape(bias_ih[hidden_size:2*hidden_size], (hidden_size,))
            b_in = tf.reshape(bias_ih[2*hidden_size:], (hidden_size,))
            
            b_hr = tf.reshape(bias_hh[:hidden_size], (hidden_size,))
            b_hz = tf.reshape(bias_hh[hidden_size:2*hidden_size], (hidden_size,))
            b_hn = tf.reshape(bias_hh[2*hidden_size:], (hidden_size,))
            
            r_t = tf.sigmoid(tf.matmul(input_t, w_ir) + tf.matmul(state_t, w_hr) + b_ir + b_hr)
            z_t = tf.sigmoid(tf.matmul(input_t, w_iz) + tf.matmul(state_t, w_hz) + b_iz + b_hz)
            n_t = tf.tanh(tf.matmul(input_t, w_in) + r_t * tf.matmul(state_t, w_hn) + b_in + b_hn)

            h_t = (1 - z_t) * n_t + z_t * state_t
            return h_t
        
        
        time_steps = input_tensor.shape[0]
        batch_size = input_tensor.shape[1]

        h_t = hidden_state[0] 
        output_list = []
        for t in range(time_steps):
            input_t = input_tensor[t]
            h_t = gru_cell(input_t, h_t, weight_ih, weight_hh, bias_ih, bias_hh)
            output_list.append(h_t)

        output = tf.stack(output_list)

        if batch_first:
            output = tf.transpose(output, perm=[1, 0, 2])

        hn = h_t
        output = output.numpy()
        hn = hn.numpy()
    
    return {"output": output, "hn": hn}

def main():
    A_TOL = 0.01
    input_size = 5
    hidden_size = 3
    seq_len = 2
    batch_size = 1

    input_data = {
        "input": np.random.rand(seq_len, batch_size, input_size).astype(np.float32),
        "h0": np.random.rand(1, batch_size, hidden_size).astype(np.float32),
        "weight_ih": np.random.rand(3 * hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.rand(3 * hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.rand(3 * hidden_size).astype(np.float32),
        "bias_hh": np.random.rand(3 * hidden_size).astype(np.float32),
        "input_size": input_size,
        "hidden_size": hidden_size
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Output results do not match"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Hidden state results do not match"

    print("Success")

if __name__ == "__main__":
    main()