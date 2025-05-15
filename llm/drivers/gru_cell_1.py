import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["hidden_state"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias = input_dict.get("bias", True)
    if bias:
        bias_ih = torch.tensor(input_dict["bias_ih"])
        bias_hh = torch.tensor(input_dict["bias_hh"])
    else:
        bias_ih = None
        bias_hh = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        if bias:
            bias_ih = bias_ih.cuda()
            bias_hh = bias_hh.cuda()
    
    result = torch.gru_cell(input_tensor, hidden_state, weight_ih, weight_hh, bias_ih, bias_hh)
    
    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result": result[0].numpy(), "result_h": result[1].numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hidden_state = tf.constant(input_dict["hidden_state"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias = input_dict.get("bias", True)
        if bias:
            bias_ih = tf.constant(input_dict["bias_ih"])
            bias_hh = tf.constant(input_dict["bias_hh"])
        else:
            bias_ih = None
            bias_hh = None

        input_size = input_tensor.shape[-1]
        hidden_size = hidden_state.shape[-1]
        
        def sigmoid(x):
            return 1 / (1 + tf.exp(-x))
        
        def tanh(x):
            return tf.tanh(x)
            
        weight_ih_r = weight_ih[:hidden_size, :]
        weight_ih_z = weight_ih[hidden_size: 2 * hidden_size, :]
        weight_ih_n = weight_ih[2 * hidden_size:, :]

        weight_hh_r = weight_hh[:hidden_size, :]
        weight_hh_z = weight_hh[hidden_size: 2 * hidden_size, :]
        weight_hh_n = weight_hh[2 * hidden_size:, :]
        
        if bias:
            bias_ih_r = bias_ih[:hidden_size]
            bias_ih_z = bias_ih[hidden_size: 2 * hidden_size]
            bias_ih_n = bias_ih[2 * hidden_size:]
            bias_hh_r = bias_hh[:hidden_size]
            bias_hh_z = bias_hh[hidden_size: 2 * hidden_size]
            bias_hh_n = bias_hh[2 * hidden_size:]
        else:
            bias_ih_r = tf.zeros([hidden_size], dtype=tf.float32)
            bias_ih_z = tf.zeros([hidden_size], dtype=tf.float32)
            bias_ih_n = tf.zeros([hidden_size], dtype=tf.float32)
            bias_hh_r = tf.zeros([hidden_size], dtype=tf.float32)
            bias_hh_z = tf.zeros([hidden_size], dtype=tf.float32)
            bias_hh_n = tf.zeros([hidden_size], dtype=tf.float32)

        def compute_gate(x, w_ix, w_hx, b_ix, b_hx):
            return tf.matmul(x, w_ix) + tf.matmul(hidden_state, w_hx) + b_ix + b_hx

        r_t = sigmoid(compute_gate(input_tensor, weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r))
        z_t = sigmoid(compute_gate(input_tensor, weight_ih_z, weight_hh_z, bias_ih_z, bias_hh_z))

        n_t = tanh(compute_gate(input_tensor, weight_ih_n, r_t * weight_hh_n, bias_ih_n, bias_hh_n))

        h_t = (1 - z_t) * n_t + z_t * hidden_state
        
        result = h_t.numpy()

    return {"result": result, "result_h": result}

def main():
    A_TOL = 0.01

    input_size = 10
    hidden_size = 20
    batch_size = 5
    
    input_data = {
        "input": np.random.rand(batch_size, input_size).astype(np.float32),
        "hidden_state": np.random.rand(batch_size, hidden_size).astype(np.float32),
        "weight_ih": np.random.rand(3 * hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.rand(3 * hidden_size, hidden_size).astype(np.float32),
        "bias": True,
        "bias_ih": np.random.rand(3 * hidden_size).astype(np.float32),
        "bias_hh": np.random.rand(3 * hidden_size).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()