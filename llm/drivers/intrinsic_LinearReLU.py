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
    
    linear = nn.Linear(weight.shape[1], weight.shape[0])
    linear.weight = nn.Parameter(weight)
    if bias is not None:
        linear.bias = nn.Parameter(bias)
    else:
        linear.bias = None
    
    relu = nn.ReLU()
    linear_relu = nn.Sequential(linear, relu)

    result = linear_relu(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

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
        
        linear_result = tf.matmul(tf.expand_dims(input_tensor, 0), weight, transpose_b=True)
        if bias is not None:
            linear_result = tf.add(linear_result, bias)
        
        result = tf.nn.relu(linear_result)
        result = tf.squeeze(result, axis=0)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985], dtype=np.float32),
        "weight": np.array([[0.5, 0.6],[0.7, 0.8]], dtype=np.float32),
        "bias": np.array([0.1, 0.2], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()