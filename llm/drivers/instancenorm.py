import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict.get("running_mean", np.array([0.0])), dtype=torch.float32)
    running_var = torch.tensor(input_dict.get("running_var", np.array([1.0])), dtype=torch.float32)
    weight = torch.tensor(input_dict.get("weight", np.array([1.0])), dtype=torch.float32)
    bias = torch.tensor(input_dict.get("bias", np.array([0.0])), dtype=torch.float32)
    use_input_stats = input_dict.get("use_input_stats", True)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    num_features = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1
    instance_norm = torch.nn.InstanceNorm1d(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=use_input_stats)

    if running_mean.shape != torch.Size([num_features]) and use_input_stats:
       instance_norm.running_mean = torch.zeros(num_features)
    elif use_input_stats:
        instance_norm.running_mean = running_mean
    if running_var.shape != torch.Size([num_features]) and use_input_stats:
        instance_norm.running_var = torch.ones(num_features)
    elif use_input_stats:
        instance_norm.running_var = running_var
    
    if weight.shape != torch.Size([num_features]):
        instance_norm.weight = torch.nn.Parameter(torch.ones(num_features))
    else:
        instance_norm.weight = torch.nn.Parameter(weight)
    
    if bias.shape != torch.Size([num_features]):
        instance_norm.bias = torch.nn.Parameter(torch.zeros(num_features))
    else:
        instance_norm.bias = torch.nn.Parameter(bias)

    if not cpu:
        input_tensor = input_tensor.cuda()
        instance_norm.cuda()
    
    result = instance_norm(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        running_mean = tf.constant(input_dict.get("running_mean", np.array([0.0])), dtype=tf.float32)
        running_var = tf.constant(input_dict.get("running_var", np.array([1.0])), dtype=tf.float32)
        weight = tf.constant(input_dict.get("weight", np.array([1.0])), dtype=tf.float32)
        bias = tf.constant(input_dict.get("bias", np.array([0.0])), dtype=tf.float32)
        use_input_stats = input_dict.get("use_input_stats", True)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        input_shape = input_tensor.shape

        if len(input_shape) == 4:
            num_features = input_shape[1]
            axes = [2, 3]
            reshape_dims = [1, num_features, 1, 1]
        elif len(input_shape) == 3:
            num_features = input_shape[1]
            axes = [1, 2]
            reshape_dims = [1, num_features, 1]
        elif len(input_shape) == 2:
            num_features = input_shape[1]
            axes = [1]
            reshape_dims = [1, num_features]
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

        if use_input_stats:
            scale = tf.cast(tf.math.rsqrt(variance + eps), tf.float32)
            offset = -mean * scale
        else:
            scale = tf.cast(tf.math.rsqrt(running_var + eps), tf.float32)
            offset = -running_mean * scale

        if len(weight.shape) > 0:
            scale = scale * tf.reshape(weight, reshape_dims)
        else:
            scale = scale * weight
        if len(bias.shape) > 0:
            offset = offset + tf.reshape(bias, reshape_dims)
        else:
            offset = offset + bias

        result = scale * input_tensor + offset
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[-0.3482,  0.8191, -0.2910, -0.0325],
         [-0.7797,  0.3625,  0.4853, -0.0033]],

        [[ 0.4828, -0.6208, -0.3065, -0.4870],
         [-0.3018,  0.0550, -0.0565,  0.2420]]], dtype=np.float32),
        "running_mean": np.array([0.3, 0.2], dtype=np.float32),
        "running_var": np.array([0.4, 0.5], dtype=np.float32),
        "weight": np.array([0.6, 0.7], dtype=np.float32),
        "bias": np.array([0.8, 0.9], dtype=np.float32),
        "use_input_stats": True,
        "momentum": 0.1,
        "eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()