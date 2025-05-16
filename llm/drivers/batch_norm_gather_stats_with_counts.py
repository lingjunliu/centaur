import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    mean_weight = torch.tensor(input_dict["mean_weight"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    counts = torch.tensor(input_dict["counts"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        mean_weight = mean_weight.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        counts = counts.cuda()

    result = torch.batch_norm_gather_stats_with_counts(
        input_tensor,
        mean_weight,
        running_mean,
        running_var,
        counts,
        momentum=momentum,
        eps=eps,
    )

    if not cpu:
        result = (result[0].cpu(), result[1].cpu())
    else:
        result = (result[0], result[1])

    return {"result_mean": result[0].numpy(), "result_var": result[1].numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        mean_weight = tf.constant(input_dict["mean_weight"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)
        counts = tf.cast(tf.constant(input_dict["counts"]), dtype=tf.float32)

        batch_mean = tf.reduce_sum(input_tensor * mean_weight, axis=0)
        batch_var = tf.reduce_sum(mean_weight * tf.math.squared_difference(input_tensor, batch_mean), axis=0)

        new_running_mean = running_mean * (1 - momentum) + batch_mean * momentum
        unbiased_var = batch_var * (tf.reduce_sum(counts) / (tf.reduce_sum(counts) - 1.0))
        new_running_var = running_var * (1 - momentum) + unbiased_var * momentum

        return {
            "result_mean": new_running_mean.numpy(),
            "result_var": new_running_var.numpy(),
        }

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "mean_weight": np.array([[0.1, 0.1], [0.2, 0.2], [0.7, 0.7]], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0], dtype=np.float32),
        "counts": np.array([1,1,1], dtype=np.int64),
        "momentum": 0.1,
        "eps": 1e-05,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result_mean"], tf_result["result_mean"], atol=A_TOL)
    assert np.allclose(torch_result["result_var"], tf_result["result_var"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()