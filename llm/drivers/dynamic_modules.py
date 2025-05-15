import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"], requires_grad=False)
    weight = torch.tensor(input_dict["weight"], requires_grad=False)
    bias = input_dict.get("bias", None)
    if bias is not None:
        bias = torch.tensor(bias, requires_grad=False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
    
    linear = nn.Linear(input_tensor.shape[-1], weight.shape[0], bias=bias is not None)
    linear.weight = torch.nn.Parameter(weight)
    if bias is not None:
        linear.bias = torch.nn.Parameter(bias)
    else:
        linear.bias = None

    with torch.no_grad():
        result = linear(input_tensor)
    
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
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias)

        result = tf.matmul(input_tensor, tf.transpose(weight))
        if bias is not None:
            result = tf.add(result, bias)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "weight": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "bias": np.array([0.01, 0.02], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "weight": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()