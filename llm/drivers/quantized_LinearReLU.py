import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
from torch.nn.intrinsic.quantized import LinearReLU
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    linear_relu = LinearReLU(input_dict["in_features"], input_dict["out_features"])
    
    linear_relu.weight = torch.nn.Parameter(torch.empty_like(linear_relu.weight.data).copy_(weight))
    if bias is not None:
        linear_relu.bias = torch.nn.Parameter(torch.empty_like(linear_relu.bias.data).copy_(bias))
    else:
        linear_relu.bias = None
    
    result = linear_relu(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None

        weight = tf.transpose(weight)
        linear = tf.matmul(input_tensor, weight)
        if bias is not None:
            linear = tf.add(linear, bias)
        
        result = tf.nn.relu(linear)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 5).astype(np.float32),
        "weight": np.random.rand(10, 5).astype(np.float32),
        "bias": np.random.rand(10).astype(np.float32),
        "in_features": 5,
        "out_features": 10
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()