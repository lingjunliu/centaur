import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict.get("num_features")
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)
    
    if num_features is None:
        num_features = input_tensor.shape[-1]
    
    bn = torch.nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if "running_mean" in input_dict:
        bn.running_mean = torch.tensor(input_dict["running_mean"])
    if "running_var" in input_dict:
        bn.running_var = torch.tensor(input_dict["running_var"])
    if "weight" in input_dict and affine:
        bn.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
    if "bias" in input_dict and affine:
        bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))

    if not cpu:
        input_tensor = input_tensor.cuda()
        bn = bn.cuda()

    result = bn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    num_features = input_dict.get("num_features")
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if num_features is None:
        num_features = input_tensor.shape[-1]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        gamma = None
        beta = None
        moving_mean = None
        moving_variance = None

        if affine:
            if "weight" in input_dict:
                gamma = tf.constant(input_dict["weight"])
            else:
                gamma = tf.ones([num_features], dtype=input_tensor.dtype)
            if "bias" in input_dict:
                beta = tf.constant(input_dict["bias"])
            else:
                beta = tf.zeros([num_features], dtype=input_tensor.dtype)

        if track_running_stats:
            if "running_mean" in input_dict:
                moving_mean = tf.constant(input_dict["running_mean"])
            else:
                moving_mean = tf.zeros([num_features], dtype=input_tensor.dtype)

            if "running_var" in input_dict:
                moving_variance = tf.constant(input_dict["running_var"])
            else:
                moving_variance = tf.ones([num_features], dtype=input_tensor.dtype)
        
        if len(input_tensor.shape) == 2:
             mean, variance = tf.nn.moments(input_tensor, axes=[0])
        else:
             mean, variance = tf.nn.moments(input_tensor, axes=[0,1])
            
        def update_running_mean(existing, new, momentum):
                return existing * momentum + new * (1 - momentum)

        def update_running_variance(existing, new, momentum):
            return existing * momentum + new * (1 - momentum)
            
        if track_running_stats:
            moving_mean = update_running_mean(moving_mean, mean, momentum)
            moving_variance = update_running_variance(moving_variance, variance, momentum)

        if affine:
            result = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)
        else:
            result = tf.nn.batch_normalization(input_tensor, mean, variance, offset=0.0, scale=1.0, variance_epsilon=eps)

        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "num_features": 3,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "running_mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()