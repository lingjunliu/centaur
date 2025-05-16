import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    count = torch.tensor(input_dict["count"])
    mean = torch.tensor(input_dict["mean"])
    var = torch.tensor(input_dict["var"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        count = count.cuda()
        mean = mean.cuda()
        var = var.cuda()
        
    torch.batch_norm_gather_stats_with_counts(
        input_tensor,
        running_mean,
        running_var,
        momentum,
        eps,
        count,
        mean,
        var
    )

    if not cpu:
        running_mean = running_mean.cpu()
        running_var = running_var.cpu()

    return {
        "running_mean": running_mean.numpy(),
        "running_var": running_var.numpy()
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        running_mean = tf.Variable(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.Variable(input_dict["running_var"], dtype=tf.float32)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)
        count = tf.constant(input_dict["count"], dtype=tf.float32)
        mean = tf.constant(input_dict["mean"], dtype=tf.float32)
        var = tf.constant(input_dict["var"], dtype=tf.float32)

        new_running_mean = (1 - momentum) * running_mean + momentum * mean
        new_running_var = (1 - momentum) * running_var + momentum * var * (count / (count - 1))
        running_mean.assign(new_running_mean)
        running_var.assign(new_running_var)
        
        rm = running_mean.numpy()
        rv = running_var.numpy()
        

    return {
        "running_mean": rm,
        "running_var": rv
    }

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "count": np.array([5], dtype=np.int64),
        "mean": np.array([3.0], dtype=np.float32),
        "var": np.array([2.0], dtype=np.float32),
        "momentum": 0.1,
        "eps": 1e-05
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["running_mean"], tf_result["running_mean"], atol=A_TOL), "Running mean results do not match"
    assert np.allclose(torch_result["running_var"], tf_result["running_var"], atol=A_TOL), "Running var results do not match"

    print("Success")

if __name__ == "__main__":
    main()