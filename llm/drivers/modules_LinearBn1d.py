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

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    freeze_bn = input_dict.get("freeze_bn", False)
    qconfig = input_dict.get("qconfig", torch.quantization.default_qconfig)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    linear_bn1d = nniqat.LinearBn1d(input_tensor.shape[1], weight.shape[0], eps=eps, momentum=momentum, freeze_bn=freeze_bn, qconfig=qconfig)

    linear_bn1d.weight = nn.Parameter(weight)
    linear_bn1d.bias = nn.Parameter(bias)
    linear_bn1d.bn.running_mean = running_mean
    linear_bn1d.bn.running_var = running_var
    linear_bn1d.bn.num_batches_tracked.data = torch.tensor(1)

    result = linear_bn1d(input_tensor)

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
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        freeze_bn = input_dict.get("freeze_bn", False)
        
        linear_output = tf.matmul(input_tensor, tf.transpose(weight)) + bias

        # Batch Normalization
        gamma = tf.ones(weight.shape[0], dtype=tf.float32)
        beta = tf.zeros(weight.shape[0], dtype=tf.float32)
        
        if freeze_bn:
            mean = running_mean
            variance = running_var
        else:
            mean, variance = tf.nn.moments(linear_output, axes=[0])

            # Update running mean and variance (Exponential Moving Average)
            running_mean = running_mean * momentum + mean * (1 - momentum)
            running_var = running_var * momentum + variance * (1 - momentum)

        normalized = (linear_output - mean) / tf.sqrt(variance + eps)
        output = gamma * normalized + beta

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    from torch.quantization import default_qconfig

    input_data = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "weight": np.random.rand(5, 3).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32),
        "running_mean": np.random.rand(5).astype(np.float32),
        "running_var": np.random.rand(5).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "freeze_bn": False,
        "qconfig": default_qconfig
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()