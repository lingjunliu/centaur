import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig, default_qconfig

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    freeze_bn = input_dict.get("freeze_bn", False)
    qconfig = input_dict.get("qconfig", default_qconfig) if input_dict.get("qconfig", None) is None else input_dict.get("qconfig")

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
    
    module = nniqat.LinearBn1d(input_dict["in_features"], input_dict["out_features"], bias is not None, eps, momentum, freeze_bn, qconfig)
    module.weight = nn.Parameter(weight)
    module.bias = nn.Parameter(bias)
    module.bn.running_mean = running_mean
    module.bn.running_var = running_var
    module.eval()
    with torch.no_grad():
        result = module(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().cpu().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

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
        freeze_bn = input_dict.get("freeze_bn", False)
        
        linear = tf.matmul(input_tensor, weight) + bias

        # TensorFlow's batch normalization uses variance instead of std
        variance = running_var
        scale = tf.math.rsqrt(variance + eps)
        output = scale * (linear - running_mean)

        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    from torch.quantization import default_qconfig
    input_data = {
        "input": np.random.rand(1, 5).astype(np.float32),
        "weight": np.random.rand(5, 5).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "running_mean": np.random.rand(5).astype(np.float32),
        "running_var": np.random.rand(5).astype(np.float32),
        "in_features": 5,
        "out_features": 5,
        "qconfig": default_qconfig
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()