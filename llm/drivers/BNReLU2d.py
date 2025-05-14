import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    bn_weight = torch.tensor(input_dict["bn_weight"])
    bn_bias = torch.tensor(input_dict["bn_bias"])
    bn_running_mean = torch.tensor(input_dict["bn_running_mean"])
    bn_running_var = torch.tensor(input_dict["bn_running_var"])
    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        bn_weight = bn_weight.cuda()
        bn_bias = bn_bias.cuda()
        bn_running_mean = bn_running_mean.cuda()
        bn_running_var = bn_running_var.cuda()
        
    bn = nn.BatchNorm2d(input_tensor.shape[1])
    bn.weight = torch.nn.Parameter(bn_weight)
    bn.bias = torch.nn.Parameter(bn_bias)
    bn.running_mean = bn_running_mean
    bn.running_var = bn_running_var
    bn.momentum = momentum
    bn.eps = eps
    bn.eval()

    with torch.no_grad():
        input_tensor = bn(input_tensor)
        result = torch.relu(input_tensor)
    
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
        bn_weight = tf.constant(input_dict["bn_weight"])
        bn_bias = tf.constant(input_dict["bn_bias"])
        bn_running_mean = tf.constant(input_dict["bn_running_mean"])
        bn_running_var = tf.constant(input_dict["bn_running_var"])
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        input_shape = input_tensor.shape
        
        scale = tf.reshape(bn_weight, (1, bn_weight.shape[0], 1, 1))
        offset = tf.reshape(bn_bias, (1, bn_bias.shape[0], 1, 1))
        
        if training:
          mean, variance = tf.nn.moments(input_tensor, axes=[0, 2, 3], keepdims=True)
          inv = tf.math.rsqrt(variance + eps)
          normalized = (input_tensor - mean) * inv
          output = scale * normalized + offset
          result = tf.nn.relu(output)
        else:
          mean = tf.reshape(bn_running_mean, (1, bn_running_mean.shape[0], 1, 1))
          variance = tf.reshape(bn_running_var, (1, bn_running_var.shape[0], 1, 1))
          inv = tf.math.rsqrt(variance + eps)
          normalized = (input_tensor - mean) * inv
          output = scale * normalized + offset
          result = tf.nn.relu(output)

        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "bn_weight": np.random.randn(3).astype(np.float32),
        "bn_bias": np.random.randn(3).astype(np.float32),
        "bn_running_mean": np.random.randn(3).astype(np.float32),
        "bn_running_var": np.random.randn(3).astype(np.float32),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()