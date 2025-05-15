import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    save_mean = input_dict.get("save_mean", False)
    save_var = input_dict.get("save_var", False)
    eps = input_dict.get("eps", 1e-05)
    exponential_average_factor = input_dict.get("exponential_average_factor", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

        result = torch.cudnn_batch_norm(
            input_tensor,
            weight,
            bias,
            running_mean,
            running_var,
            training=False,
            exponential_average_factor=exponential_average_factor,
            epsilon=eps
        )

        result = result[0].cpu(), result[1].cpu(), result[2].cpu()
    else:
        mean = running_mean
        variance = running_var

        scale = weight
        offset = bias

        inv = torch.rsqrt(variance + eps)
        scaled_input = (input_tensor - mean) * inv * scale + offset

        result = (scaled_input, running_mean, running_var)


    return {
        "result": result[0].numpy(),
        "running_mean": result[1].numpy(),
        "running_var": result[2].numpy()
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
    running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
    weight = tf.constant(input_dict["weight"], dtype=tf.float32)
    bias = tf.constant(input_dict["bias"], dtype=tf.float32)
    eps = input_dict.get("eps", 1e-05)
    exponential_average_factor = input_dict.get("exponential_average_factor", 0.0)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):
        mean = running_mean
        variance = running_var

        scale = weight
        offset = bias

        inv = tf.math.rsqrt(variance + eps)
        scaled_input = (input_tensor - mean) * inv * scale + offset

        result = scaled_input.numpy()

    return {
        "result": result,
        "running_mean": running_mean.numpy(),
        "running_var": running_var.numpy()
    }

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "eps": 1e-05
    }

    torch_result = torch_version(input_data, cpu=True)
    tf_result = tensorflow_version(input_data, cpu=True)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()