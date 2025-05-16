import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    scale_factor = input_dict.get("scale_factor", None)
    zero_point = input_dict.get("zero_point", None)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        
    if scale_factor is not None:
        scale_factor = torch.tensor(scale_factor)
        if not cpu:
            scale_factor = scale_factor.cuda()
    if zero_point is not None:
        zero_point = torch.tensor(zero_point)
        if not cpu:
            zero_point = zero_point.cuda()
    else:
        zero_point = torch.tensor([0]) 
        if not cpu:
            zero_point = zero_point.cuda()

    bn_relu = torch.nn.intrinsic.quantized.BNReLU2d(input_tensor.shape[1])
    bn_relu.weight = torch.nn.Parameter(weight)
    bn_relu.bias = torch.nn.Parameter(bias)
    bn_relu.running_mean = running_mean
    bn_relu.running_var = running_var
    bn_relu.eps = eps
    bn_relu.momentum = momentum
    
    bn_relu.scale_factor = scale_factor
    bn_relu.zero_point = zero_point
    
    input_tensor = torch.quantize_per_tensor(input_tensor, scale=torch.tensor([1.0]), zero_point=torch.tensor([0]), dtype=torch.quint8)

    bn_relu.scale_factor = torch.tensor([1.0])
    bn_relu.zero_point = torch.tensor([0])
    if not cpu:
        bn_relu.scale_factor = bn_relu.scale_factor.cuda()
        bn_relu.zero_point = bn_relu.zero_point.cuda()

    result = bn_relu(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.dequantize().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        scale_factor_np = input_dict.get("scale_factor", None)
        zero_point_np = input_dict.get("zero_point", None)

        if scale_factor_np is not None:
            scale_factor = tf.constant(scale_factor_np)
        else:
            scale_factor = None
        if zero_point_np is not None:
            zero_point = tf.constant(zero_point_np)
        else:
            zero_point = tf.constant([0], dtype=tf.int32)

        mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2])

        x = (input_tensor - running_mean) / tf.sqrt(running_var + eps)
        x = x * weight + bias
        x = tf.nn.relu(x)

        result = x.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "scale_factor": np.array([1.0]).astype(np.float32),
        "zero_point": np.array([0]).astype(np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()