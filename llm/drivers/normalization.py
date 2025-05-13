import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if affine:
        weight = torch.tensor(input_dict["weight"])
        bias = torch.tensor(input_dict["bias"])
        if not cpu:
            weight = weight.cuda()
            bias = bias.cuda()

    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])

    if not cpu:
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    num_features = input_tensor.size(1) if len(input_tensor.shape) > 1 else 1
    
    if running_mean.numel() != num_features:
        running_mean = torch.zeros(num_features)
        running_var = torch.ones(num_features)

    if not cpu:
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        
    if affine:
        result = torch.nn.functional.batch_norm(
            input_tensor, running_mean, running_var, weight, bias, training=False, momentum=momentum, eps=eps
        )
    else:
        result = torch.nn.functional.batch_norm(
            input_tensor, running_mean, running_var, None, None, training=False, momentum=momentum, eps=eps
        )
    
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
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)

        if affine:
            weight = tf.constant(input_dict["weight"])
            bias = tf.constant(input_dict["bias"])

        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])

        inv = tf.math.rsqrt(running_var + eps)
        if affine:
            scaled_input = (input_tensor - running_mean) * inv * weight + bias
        else:
            scaled_input = (input_tensor - running_mean) * inv

        result = scaled_input.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()