import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-5)
    count = torch.tensor(input_dict["count"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        count = count.cuda()

    result = torch.batch_norm_gather_stats(
        input_tensor, running_mean, running_var, momentum, eps, count
    )

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


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
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-5)
        count = tf.constant(input_dict["count"], dtype=tf.float32)
        
        mean_input = tf.reduce_mean(input_tensor)
        var_input = tf.math.reduce_variance(input_tensor)
        
        new_mean = running_mean * (1 - momentum) + momentum * mean_input
        new_var = running_var * (1 - momentum) + momentum * var_input
        
        result = tf.stack([new_mean, new_var])
        
        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "momentum": 0.1,
        "eps": 1e-5,
        "count": np.array([10.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(
        torch_result["result"], tf_result["result"], atol=A_TOL
    ), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()