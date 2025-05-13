import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    qlinearrelu = torch.nn.intrinsic.quantized.modules.LinearReLU(input_dict["in_features"], input_dict["out_features"])
    
    qlinearrelu.weight = nn.Parameter(weight)
    qlinearrelu.bias = nn.Parameter(bias)
    
    qlinearrelu.scale = scale
    qlinearrelu.zero_point = zero_point
    result = qlinearrelu(input_tensor)

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
        bias = tf.constant(input_dict["bias"])
        scale = input_dict.get("scale", 1.0)
        zero_point = input_dict.get("zero_point", 0)
        
        weight_transposed = tf.transpose(weight)
        linear_output = tf.matmul(input_tensor, weight_transposed) + bias
        relu_output = tf.nn.relu(linear_output)

        result = relu_output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985]], dtype=np.float32),
        "weight": np.array([[1.1901, 0.6778], [0.8161, 2.3814]], dtype=np.float32),
        "bias": np.array([0.8778, -0.8964], dtype=np.float32),
        "in_features": 2,
        "out_features": 2,
        "scale": 1.0,
        "zero_point": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()