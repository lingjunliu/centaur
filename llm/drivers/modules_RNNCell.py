import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias = input_dict.get("bias", True)
    if "bias_ih" in input_dict and "bias_hh" in input_dict:
        bias_ih = torch.tensor(input_dict["bias_ih"])
        bias_hh = torch.tensor(input_dict["bias_hh"])
    else:
        bias_ih = None
        bias_hh = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        if bias_ih is not None:
            bias_ih = bias_ih.cuda()
        if bias_hh is not None:
            bias_hh = bias_hh.cuda()

    cell = torch.nn.RNNCell(input_size=input_tensor.shape[1], hidden_size=hx.shape[1], bias=bias)
    cell.weight_ih = torch.nn.Parameter(weight_ih)
    cell.weight_hh = torch.nn.Parameter(weight_hh)
    if bias_ih is not None and bias_hh is not None:
        cell.bias_ih = torch.nn.Parameter(bias_ih)
        cell.bias_hh = torch.nn.Parameter(bias_hh)
    else:
        cell.bias_ih = None
        cell.bias_hh = None

    result = cell(input_tensor, hx)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hx = tf.constant(input_dict["hx"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias = input_dict.get("bias", True)

        if "bias_ih" in input_dict and "bias_hh" in input_dict:
            bias_ih = tf.constant(input_dict["bias_ih"])
            bias_hh = tf.constant(input_dict["bias_hh"])
        else:
            bias_ih = None
            bias_hh = None

        Wh = weight_hh
        Wx = weight_ih
        
        h = hx
        x = input_tensor

        if bias:
            bh = bias_hh
            bx = bias_ih
            h = tf.tanh(tf.matmul(x, tf.transpose(Wx)) + bx + tf.matmul(h, tf.transpose(Wh)) + bh)
        else:
            h = tf.tanh(tf.matmul(x, tf.transpose(Wx)) + tf.matmul(h, tf.transpose(Wh)))

        result = h.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_size = 5
    hidden_size = 10
    batch_size = 3

    input_data = {
        "input": np.random.randn(batch_size, input_size).astype(np.float32),
        "hx": np.random.randn(batch_size, hidden_size).astype(np.float32),
        "weight_ih": np.random.randn(hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.randn(hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.randn(hidden_size).astype(np.float32),
        "bias_hh": np.random.randn(hidden_size).astype(np.float32),
        "bias": True
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()