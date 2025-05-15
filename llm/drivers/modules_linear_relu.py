import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    from torch.nn.intrinsic import quantized

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    model = quantized.modules.linear_relu.LinearReLU(input_dict["in_features"], input_dict["out_features"])
    
    with torch.no_grad():
        model.weight.copy_(weight)
        if bias is not None:
            model.bias.copy_(bias)

        result = model(input_tensor)

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

        linear_output = tf.matmul(input_tensor, weight, transpose_b = True)

        if bias is not None:
            linear_output = tf.add(linear_output, bias)

        result = tf.nn.relu(linear_output)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "weight": np.array([[0.5, 0.7], [0.6, 0.8]], dtype=np.float32),
        "bias": np.array([0.1, 0.2], dtype=np.float32),
        "in_features": 2,
        "out_features": 2,
    }

    model = quantized.modules.linear_relu.LinearReLU(input_data["in_features"], input_data["out_features"])
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()