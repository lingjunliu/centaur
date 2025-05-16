import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic.qat as nniqat
    from torch.quantization import QConfig
    from torch.quantization import default_observer
    from torch.nn import functional as F

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    qconfig = input_dict.get("qconfig", QConfig(activation=default_observer, weight=default_observer))

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
    
    module = nniqat.LinearBn1d(input_tensor.shape[1], weight.shape[0], qconfig=qconfig)
    module.weight = nn.Parameter(weight)
    module.bias = nn.Parameter(bias)
    module.bn.running_mean = running_mean
    module.bn.running_var = running_var
    module.bn.eps = eps
    module.bn.momentum = momentum

    if not cpu:
        module = module.cuda()
    
    result = module(input_tensor)

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
        weight = tf.constant(input_dict["weight"].transpose(), dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        
        x = tf.matmul(input_tensor, weight) + bias
        
        scale = tf.math.rsqrt(running_var + eps)
        
        result = (x - running_mean) * scale

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_dim = 3
    output_dim = 4
    batch_size = 2

    input_data = {
        "input": np.random.rand(batch_size, input_dim).astype(np.float32),
        "weight": np.random.rand(output_dim, input_dim).astype(np.float32),
        "bias": np.random.rand(output_dim).astype(np.float32),
        "running_mean": np.random.rand(output_dim).astype(np.float32),
        "running_var": np.abs(np.random.rand(output_dim)).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()