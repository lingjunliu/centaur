import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict.get("num_features", input_tensor.shape[-1])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    bn = torch.nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    if 'running_mean' in input_dict:
      bn.running_mean = torch.tensor(input_dict['running_mean'])
      if not cpu:
          bn.running_mean = bn.running_mean.cuda()
    if 'running_var' in input_dict:
      bn.running_var = torch.tensor(input_dict['running_var'])
      if not cpu:
          bn.running_var = bn.running_var.cuda()
    if 'weight' in input_dict and affine:
      bn.weight = torch.nn.Parameter(torch.tensor(input_dict['weight']))
      if not cpu:
          bn.weight = torch.nn.Parameter(bn.weight.cuda())
    if 'bias' in input_dict and affine:
      bn.bias = torch.nn.Parameter(torch.tensor(input_dict['bias']))
      if not cpu:
          bn.bias = torch.nn.Parameter(bn.bias.cuda())
        
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
        num_features = input_dict.get("num_features", input_tensor.shape[-1])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)
        
        gamma = None
        beta = None
        moving_mean = None
        moving_variance = None
        
        if 'weight' in input_dict and affine:
            gamma = tf.constant(input_dict['weight'], dtype=input_tensor.dtype)
        else:
            gamma = tf.ones([num_features], dtype=input_tensor.dtype)

        if 'bias' in input_dict and affine:
            beta = tf.constant(input_dict['bias'], dtype=input_tensor.dtype)
        else:
            beta = tf.zeros([num_features], dtype=input_tensor.dtype)
            
        if 'running_mean' in input_dict:
            moving_mean = tf.constant(input_dict['running_mean'], dtype=input_tensor.dtype)
        else:
            moving_mean = tf.zeros([num_features], dtype=input_tensor.dtype)
            
        if 'running_var' in input_dict:
            moving_variance = tf.constant(input_dict['running_var'], dtype=input_tensor.dtype)
        else:
            moving_variance = tf.ones([num_features], dtype=input_tensor.dtype)
            

        mean, variance = tf.nn.moments(input_tensor, axes=[0])
            
        if track_running_stats:            
            def update_mean_var():
              new_moving_mean = moving_mean * momentum + mean * (1 - momentum)
              new_moving_variance = moving_variance * momentum + variance * (1 - momentum)
              return new_moving_mean, new_moving_variance
            
            moving_mean, moving_variance = update_mean_var()

        result = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056],
                           [0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "num_features": 4,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "running_mean": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "running_var": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()