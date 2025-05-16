import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    grad_output = torch.tensor(input_dict["grad_output"])
    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    running_mean = torch.tensor(input_dict["running_mean"])
    var = torch.tensor(input_dict["var"])
    eps = input_dict.get("eps", 1e-05)
    save_mean = input_dict.get("save_mean", False)
    save_var = input_dict.get("save_var", False)

    if not cpu:
        grad_output = grad_output.cuda()
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        running_mean = running_mean.cuda()
        var = var.cuda()

    result = torch.batch_norm_backward_reduce(
        grad_output,
        input_tensor,
        weight,
        running_mean,
        var,
        save_mean,
        save_var,
        eps
    )

    if not cpu:
        result = [r.cpu() for r in result]

    return {
        "sum_dy": result[0].numpy(),
        "sum_dy_xmu": result[1].numpy(),
        "grad_weight": result[2].numpy(),
        "grad_bias": result[3].numpy(),
    }


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    grad_output = tf.constant(input_dict["grad_output"])
    input_tensor = tf.constant(input_dict["input"])
    weight = tf.constant(input_dict["weight"])
    running_mean = tf.constant(input_dict["running_mean"])
    var = tf.constant(input_dict["var"])
    eps = input_dict.get("eps", 1e-05)
    save_mean = input_dict.get("save_mean", False)
    save_var = input_dict.get("save_var", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        mean = running_mean
        inv_std = 1.0 / tf.sqrt(var + eps)
        x_hat = (input_tensor - mean) * inv_std

        sum_dy = tf.reduce_sum(grad_output)
        sum_dy_xmu = tf.reduce_sum(grad_output * x_hat)
        grad_weight = tf.reduce_sum(grad_output * x_hat) * weight
        grad_bias = tf.reduce_sum(grad_output)

    return {
        "sum_dy": sum_dy.numpy(),
        "sum_dy_xmu": sum_dy_xmu.numpy(),
        "grad_weight": grad_weight.numpy(),
        "grad_bias": grad_bias.numpy(),
    }

def main():
    A_TOL = 0.01

    input_data = {
        "grad_output": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "weight": np.array([0.5], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "var": np.array([1.0], dtype=np.float32),
        "eps": 1e-05,
        "save_mean": False,
        "save_var": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["sum_dy"], tf_result["sum_dy"], atol=A_TOL), "sum_dy mismatch"
    assert np.allclose(torch_result["sum_dy_xmu"], tf_result["sum_dy_xmu"], atol=A_TOL), "sum_dy_xmu mismatch"
    assert np.allclose(torch_result["grad_weight"], tf_result["grad_weight"], atol=A_TOL), "grad_weight mismatch"
    assert np.allclose(torch_result["grad_bias"], tf_result["grad_bias"], atol=A_TOL), "grad_bias mismatch"

    print("Success")

if __name__ == "__main__":
    main()