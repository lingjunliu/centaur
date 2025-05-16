import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    bias = input_dict.get("bias", True)
    nonlinearity = input_dict.get("nonlinearity", "tanh")
    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    hidden_tensor = torch.tensor(input_dict.get("hidden", np.zeros((input_tensor.shape[0], hidden_size))), dtype=torch.float32)

    rnn = torch.nn.RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
    
    if not cpu:
        rnn = rnn.cuda()
        input_tensor = input_tensor.cuda()
        hidden_tensor = hidden_tensor.cuda()

    with torch.no_grad():
        h = rnn(input_tensor, hidden_tensor)

        if not cpu:
            h = h.cpu()

    return {"output": h.cpu().numpy()} if not cpu else {"output": h.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    bias = input_dict.get("bias", True)
    nonlinearity = input_dict.get("nonlinearity", "tanh")
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    hidden_tensor = tf.constant(input_dict.get("hidden", np.zeros((input_tensor.shape[0], hidden_size))), dtype=tf.float32)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        W_ih = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size), size=(hidden_size, input_size)).astype(np.float32), dtype=tf.float32)
        W_hh = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size), size=(hidden_size, hidden_size)).astype(np.float32), dtype=tf.float32)

        if bias:
            b_ih = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size), size=(hidden_size,)).astype(np.float32), dtype=tf.float32)
            b_hh = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size), size=(hidden_size,)).astype(np.float32), dtype=tf.float32)

        x = tf.matmul(input_tensor, W_ih, transpose_b=True)
        h = tf.matmul(hidden_tensor, W_hh, transpose_b=True)

        if bias:
            x = tf.add(x, b_ih)
            h = tf.add(h, b_hh)

        if nonlinearity == "tanh":
            out = tf.tanh(tf.add(x, h))
        elif nonlinearity == "relu":
            out = tf.nn.relu(tf.add(x, h))
        else:
            raise ValueError("Nonlinearity must be 'tanh' or 'relu'")

        out = out.numpy()

    return {"output": out}

def main():
    A_TOL = 0.01

    input_size = 10
    hidden_size = 20
    batch_size = 3

    input_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "input": np.random.randn(batch_size, input_size).astype(np.float32),
        "hidden": np.random.randn(batch_size, hidden_size).astype(np.float32),
        "bias": True,
        "nonlinearity": "relu"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()