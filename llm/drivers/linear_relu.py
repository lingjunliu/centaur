import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    from torch.nn.intrinsic import quantized

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    model = quantized.modules.linear_relu.LinearReLU(input_tensor.shape[1], weight.shape[0])
    with torch.no_grad():
        model.weight[:] = weight
        if bias is not None:
            model.bias[:] = bias

    result = model(input_tensor)

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
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None

        linear = tf.matmul(input_tensor, tf.transpose(weight))
        if bias is not None:
            linear = tf.add(linear, bias)
        result = tf.nn.relu(linear)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 5).astype(np.float32),
        "weight": np.random.rand(3, 5).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()