import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    eps = input_dict.get("eps", 1e-05)
    exponential_average_factor = input_dict.get("exponential_average_factor", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    num_features = input_tensor.shape[1]
    if cpu:
      bn = torch.nn.BatchNorm2d(num_features)
      bn.weight = torch.nn.Parameter(weight)
      bn.bias = torch.nn.Parameter(bias)
      bn.running_mean = running_mean
      bn.running_var = running_var
      bn.eps = eps
      bn.momentum = exponential_average_factor
      result = bn(input_tensor)
    else:
      bn = torch.nn.BatchNorm2d(num_features).cuda()
      bn.weight = torch.nn.Parameter(weight).cuda()
      bn.bias = torch.nn.Parameter(bias).cuda()
      bn.running_mean = running_mean.cuda()
      bn.running_var = running_var.cuda()
      bn.eps = eps
      bn.momentum = exponential_average_factor
      result = bn(input_tensor)

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
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        exponential_average_factor = input_dict.get("exponential_average_factor", 0.0)
        training = True

        result = tf.nn.batch_normalization(input_tensor, running_mean, running_var, offset=bias, scale=weight, variance_epsilon=eps)

        return {"result": result.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "eps": 1e-05,
        "exponential_average_factor": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()