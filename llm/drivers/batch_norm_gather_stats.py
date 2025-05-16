import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()


def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    bns = torch.tensor(input_dict["bns"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    count = torch.tensor(input_dict["count"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        bns = bns.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        count = count.cuda()

    result_mean, result_var = torch.batch_norm_gather_stats(
        input_tensor,
        bns,
        running_mean,
        running_var,
        momentum=momentum,
        eps=eps,
        count=count
    )

    if not cpu:
        result_mean = result_mean.cpu()
        result_var = result_var.cpu()

    return {"result_mean": result_mean.numpy(), "result_var": result_var.numpy()}


def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device = "/cpu:0"
    else:
        device = "/gpu:0"

    with tf.device(device):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        bns = tf.constant(input_dict["bns"], dtype=tf.int32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)
        count = tf.constant(input_dict["count"], dtype=tf.float32)

        bns_indices = bns
        gathered_mean = tf.gather(running_mean, bns_indices)
        gathered_var = tf.gather(running_var, bns_indices)

        axes = list(range(len(input_tensor.shape) - 1))
        if not axes:
            axes = 0

        batch_mean, batch_variance = tf.nn.moments(input_tensor, axes=axes)

        if count > 0:
            new_running_mean = (1 - momentum) * gathered_mean + momentum * batch_mean
            unbiased_var = count / (count - 1) * batch_variance if count > 1 else batch_variance
            new_running_var = (1 - momentum) * gathered_var + momentum * unbiased_var
        else:
            new_running_mean = batch_mean
            new_running_var = batch_variance

        result_mean = new_running_mean.numpy()
        result_var = new_running_var.numpy()

    return {"result_mean": result_mean, "result_var": result_var}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.4033, 0.3663, -2.5241, -1.4478],
                           [0.8452, -0.3443, -0.0603, -0.0832]], dtype=np.float32),
        "bns": np.array([0, 1, 0], dtype=np.int64),
        "running_mean": np.array([0.4877, -0.3269, -1.1242, 0.3806], dtype=np.float32),
        "running_var": np.array([1.8644, 0.0987, 1.0615, 1.0647], dtype=np.float32),
        "momentum": 0.1,
        "eps": 1e-05,
        "count": np.array(2, dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result_mean"], tf_result["result_mean"], atol=A_TOL), "Mean results do not match"
    assert np.allclose(torch_result["result_var"], tf_result["result_var"], atol=A_TOL), "Variance results do not match"

    print("Success")


if __name__ == "__main__":
    main()