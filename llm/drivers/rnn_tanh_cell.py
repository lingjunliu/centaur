import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
    bias_hh = torch.tensor(input_dict["bias_hh"]) if "bias_hh" in input_dict else None
    hx = torch.tensor(input_dict["hx"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        if bias_ih is not None:
            bias_ih = bias_ih.cuda()
        if bias_hh is not None:
            bias_hh = bias_hh.cuda()
        hx = hx.cuda()

    if bias_ih is not None and bias_hh is not None:
        result = torch.tanh(torch.mm(input_tensor, weight_ih.t()) + bias_ih + torch.mm(hx, weight_hh.t()) + bias_hh)
    else:
        result = torch.tanh(torch.mm(input_tensor, weight_ih.t()) + torch.mm(hx, weight_hh.t()))

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = tf.constant(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
        bias_hh = tf.constant(input_dict["bias_hh"]) if "bias_hh" in input_dict else None
        hx = tf.constant(input_dict["hx"])

        if bias_ih is not None and bias_hh is not None:
            result = tf.tanh(tf.matmul(input_tensor, tf.transpose(weight_ih)) + bias_ih + tf.matmul(hx, tf.transpose(weight_hh)) + bias_hh)
        else:
            result = tf.tanh(tf.matmul(input_tensor, tf.transpose(weight_ih)) + tf.matmul(hx, tf.transpose(weight_hh)))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2]], dtype=np.float32),
        "weight_ih": np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        "weight_hh": np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
        "bias_ih": np.array([0.1, 0.2], dtype=np.float32),
        "bias_hh": np.array([0.3, 0.4], dtype=np.float32),
        "hx": np.array([[0.2, 0.3]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()