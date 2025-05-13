import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = nn.Linear(input_tensor.shape[-1], weight.shape[0])
    result.weight = torch.nn.Parameter(weight)
    if bias is not None:
        result.bias = torch.nn.Parameter(bias)
    else:
        result.bias = None
    if not cpu:
        result = result.cuda()
    with torch.no_grad():
        result = result(input_tensor)

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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None

        result = tf.matmul(input_tensor, tf.transpose(weight))
        if bias is not None:
            result = tf.add(result, bias)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "weight": np.array([[0.7645, 0.8335, 0.3677, -0.2633],
                            [0.8670, 0.6348, 0.5393, -0.6113]], dtype=np.float32),
        "bias": np.array([0.5306, -0.4142], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()