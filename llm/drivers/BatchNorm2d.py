import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    bn = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    if "weight" in input_dict and "bias" in input_dict:
      bn.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
      bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))
    
    if "running_mean" in input_dict and "running_var" in input_dict:
      bn.running_mean = torch.tensor(input_dict["running_mean"])
      bn.running_var = torch.tensor(input_dict["running_var"])
    
    result = bn(input_tensor)
    
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
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)
        
        axes = [0, 2, 3]
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

        if 'running_mean' in input_dict and 'running_var' in input_dict and track_running_stats:
            running_mean = tf.constant(input_dict['running_mean'], dtype=tf.float32)
            running_var = tf.constant(input_dict['running_var'], dtype=tf.float32)

            new_running_mean = running_mean * momentum + mean * (1 - momentum)
            new_running_var = running_var * momentum + variance * (1 - momentum)

            mean = running_mean
            variance = running_var

        def broadcast_parameter(param):
          shape = [1, num_features, 1, 1]
          return tf.reshape(param, shape)
            
        if affine:
            weight = tf.constant(input_dict["weight"], dtype=tf.float32)
            bias = tf.constant(input_dict["bias"], dtype=tf.float32)
            
            weight = broadcast_parameter(weight)
            bias = broadcast_parameter(bias)
            
            result = tf.nn.batch_normalization(input_tensor, mean, variance, bias, weight, eps)
        else:
            result = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, eps)
    
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "num_features": 3,
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()