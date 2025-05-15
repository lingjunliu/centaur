import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict.get("num_features")
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", False)
    track_running_stats = input_dict.get("track_running_stats", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    instance_norm = torch.nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if not cpu:
        instance_norm = instance_norm.cuda()

    if affine:
        instance_norm.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
        instance_norm.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))
        if not cpu:
            instance_norm.weight = instance_norm.weight.cuda()
            instance_norm.bias = instance_norm.bias.cuda()
        
    if track_running_stats:
        instance_norm.running_mean = torch.tensor(input_dict["running_mean"])
        instance_norm.running_var = torch.tensor(input_dict["running_var"])
        if not cpu:
            instance_norm.running_mean = instance_norm.running_mean.cuda()
            instance_norm.running_var = instance_norm.running_var.cuda()
        
    result = instance_norm(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        num_features = input_dict.get("num_features")
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", False)
        track_running_stats = input_dict.get("track_running_stats", False)

        input_shape = input_tensor.shape
        axes = [i for i in range(1, len(input_shape)-1)]
        
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)
        
        if track_running_stats:
            running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
            running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
            
            def update_running_stats():
                new_running_mean = momentum * mean + (1 - momentum) * tf.reshape(running_mean, [1, num_features, 1, 1])
                new_running_var = momentum * variance + (1 - momentum) * tf.reshape(running_var, [1, num_features, 1, 1])
                return new_running_mean, new_running_var
            
            new_running_mean, new_running_var = update_running_stats()
            
            running_mean = new_running_mean
            running_var = new_running_var

        normalized_tensor = tf.nn.batch_normalization(input_tensor, mean, variance, offset=None, scale=None, variance_epsilon=eps)

        if affine:
            weight = tf.constant(input_dict["weight"], dtype=tf.float32)
            bias = tf.constant(input_dict["bias"], dtype=tf.float32)

            normalized_tensor = normalized_tensor * tf.reshape(weight, (1, num_features, 1, 1)) + tf.reshape(bias, (1, num_features, 1, 1))

        result = normalized_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 32, 32).astype(np.float32),
        "num_features": 3,
        "affine": True,
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "track_running_stats": True,
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "eps": 1e-5,
        "momentum": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()