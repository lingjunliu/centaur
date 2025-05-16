import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    result, save_mean, save_var = torch.native_batch_norm(
        input_tensor,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )

    if not cpu:
        result = result.cpu()
        save_mean = save_mean.cpu()
        save_var = save_var.cpu()

    return {"result": result.numpy(), "save_mean": save_mean.numpy(), "save_var": save_var.numpy()}


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
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else tf.zeros(input_dict["weight"].shape[0], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        input_shape = input_tensor.shape
        if len(input_shape) == 2:
            axes = [0]
            size = input_shape[0]
        elif len(input_shape) == 4:
            axes = [0, 1, 2]
            size = input_shape[0] * input_shape[1] * input_shape[2]
        else:
             raise ValueError("Input tensor must be 2D or 4D")

        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)

        if training:
            result = tf.nn.batch_normalization(
                input_tensor, mean, variance, bias, weight, eps
            )
            
            # Corrected unbiasing of variance
            unbiased_variance = variance * (size / (size - 1))

            new_running_mean = (1 - momentum) * running_mean + momentum * mean
            new_running_var = (1 - momentum) * running_var + momentum * unbiased_variance


            save_mean = new_running_mean.numpy()
            save_var = new_running_var.numpy()
        else:
            result = tf.nn.batch_normalization(
                input_tensor, running_mean, running_var, bias, weight, eps
            )
            save_mean = running_mean.numpy()
            save_var = running_var.numpy()

        result = result.numpy()

    return {"result": result, "save_mean": save_mean, "save_var": save_var}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
        "weight": np.array([1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0], dtype=np.float32),
        "training": True,
        "momentum": 0.1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Result Mismatch"
    assert np.allclose(torch_result["save_mean"], tf_result["save_mean"], atol=A_TOL), "save_mean Mismatch"
    assert np.allclose(torch_result["save_var"], tf_result["save_var"], atol=A_TOL), "save_var Mismatch"

    print("Success")


if __name__ == "__main__":
    main()