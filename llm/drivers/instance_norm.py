import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict.get("running_mean", np.array([0.0])), dtype=torch.float32)
    running_var = torch.tensor(input_dict.get("running_var", np.array([1.0])), dtype=torch.float32)
    weight = torch.tensor(input_dict.get("weight", np.array([1.0])), dtype=torch.float32)
    bias = torch.tensor(input_dict.get("bias", np.array([0.0])), dtype=torch.float32)
    use_input_stats = input_dict.get("use_input_stats", True)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    num_features = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1
    
    if len(input_tensor.shape) == 4:
        instance_norm = torch.nn.InstanceNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=use_input_stats)
    elif len(input_tensor.shape) == 3:
        instance_norm = torch.nn.InstanceNorm1d(num_features=num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=use_input_stats)
    elif len(input_tensor.shape) == 2:
        instance_norm = torch.nn.InstanceNorm1d(num_features=num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=use_input_stats)
    else:
        instance_norm = torch.nn.InstanceNorm1d(num_features=num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=use_input_stats)


    instance_norm.weight = torch.nn.Parameter(weight)
    instance_norm.bias = torch.nn.Parameter(bias)
    instance_norm.running_mean = running_mean
    instance_norm.running_var = running_var
    instance_norm.eval() 
    
    with torch.no_grad():
        result = instance_norm(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        running_mean = tf.constant(input_dict.get("running_mean", np.array([0.0])), dtype=tf.float32)
        running_var = tf.constant(input_dict.get("running_var", np.array([1.0])), dtype=tf.float32)
        weight = tf.constant(input_dict.get("weight", np.array([1.0])), dtype=tf.float32)
        bias = tf.constant(input_dict.get("bias", np.array([0.0])), dtype=tf.float32)
        use_input_stats = input_dict.get("use_input_stats", True)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        input_shape = input_tensor.shape
        num_features = input_shape[-1]
        axes = list(range(len(input_shape) - 1))

        if use_input_stats:
            mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)
        else:
            mean = tf.reshape(running_mean, (1,)*(len(input_shape)-1) + (num_features,))
            variance = tf.reshape(running_var, (1,)*(len(input_shape)-1) + (num_features,))

        result = tf.nn.batch_normalization(
            input_tensor,
            mean,
            variance,
            bias,
            weight,
            eps
        )
        
        result = result.numpy()
        
    return {"result": result}


def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[[0.0202, 1.0985], [1.3506, -0.6056]]]], dtype=np.float32),
        "weight": np.array([1.0], dtype=np.float32),
        "bias": np.array([0.0], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "use_input_stats": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()