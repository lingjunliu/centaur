import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.quantized.modules import BNReLU2d

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if affine:
        weight = torch.tensor(input_dict["weight"])
        bias = torch.tensor(input_dict["bias"])
    else:
        weight = None
        bias = None

    if track_running_stats:
        running_mean = torch.tensor(input_dict["running_mean"])
        running_var = torch.tensor(input_dict["running_var"])
    else:
        running_mean = None
        running_var = None
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if affine:
            weight = weight.cuda()
            bias = bias.cuda()
        if track_running_stats:
            running_mean = running_mean.cuda()
            running_var = running_var.cuda()

    bn_relu = BNReLU2d(num_features, eps, momentum, affine, track_running_stats)

    if affine and bn_relu.weight is not None and bn_relu.bias is not None:
        with torch.no_grad():
            bn_relu.weight.copy_(weight)
            bn_relu.bias.copy_(bias)
    
    if track_running_stats and bn_relu.running_mean is not None and bn_relu.running_var is not None:
        with torch.no_grad():
            bn_relu.running_mean.copy_(running_mean)
            bn_relu.running_var.copy_(running_var)

    result = bn_relu(input_tensor.float())
    
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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        if affine:
            weight = tf.convert_to_tensor(input_dict["weight"], dtype=tf.float32)
            bias = tf.convert_to_tensor(input_dict["bias"], dtype=tf.float32)
        else:
            weight = tf.ones([num_features], dtype=tf.float32)
            bias = tf.zeros([num_features], dtype=tf.float32)
        
        if track_running_stats:
            running_mean = tf.convert_to_tensor(input_dict["running_mean"], dtype=tf.float32)
            running_var = tf.convert_to_tensor(input_dict["running_var"], dtype=tf.float32)
        else:
            running_mean = tf.zeros([num_features], dtype=tf.float32)
            running_var = tf.ones([num_features], dtype=tf.float32)

        mean = running_mean
        variance = running_var

        x = input_tensor

        scale = weight * tf.math.rsqrt(variance + eps)
        y = scale * (x - mean) + bias
        y = tf.nn.relu(y)
        result = y
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_shape = (1, 2, 3, 3)
    num_features = input_shape[1]

    input_data = {
        "input": np.random.rand(*input_shape).astype(np.float32),
        "num_features": num_features,
        "weight": np.random.rand(num_features).astype(np.float32),
        "bias": np.random.rand(num_features).astype(np.float32),
        "running_mean": np.random.rand(num_features).astype(np.float32),
        "running_var": np.random.rand(num_features).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()