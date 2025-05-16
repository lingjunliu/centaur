import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.intrinsic as nni

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    linear = nn.Linear(input_dict["input"].shape[1], input_dict["weight"].shape[0])
    bn = nn.BatchNorm1d(input_dict["weight"].shape[0])
    linear_bn = nni.LinearBn1d(linear, bn)

    linear_bn.weight = nn.Parameter(weight)
    linear_bn.bias = nn.Parameter(bias)
    linear_bn.running_mean = running_mean
    linear_bn.running_var = running_var
    linear_bn.eps = eps
    linear_bn.momentum = momentum

    result = linear_bn(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)

        linear = tf.matmul(input_tensor, tf.transpose(weight)) + bias
        
        result = tf.raw_ops.BatchNormTraining(
            x=linear,
            mean=running_mean,
            variance=running_var,
            offset=bias,
            scale=tf.ones_like(bias),
            variance_epsilon=eps
        )[0]

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "weight": np.random.rand(4, 3).astype(np.float32),
        "bias": np.random.rand(4).astype(np.float32),
        "running_mean": np.random.rand(4).astype(np.float32),
        "running_var": np.random.rand(4).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()