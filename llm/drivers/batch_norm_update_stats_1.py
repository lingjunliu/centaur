import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)
        running_mean = running_mean.unsqueeze(0)
        running_var = running_var.unsqueeze(0)
        torch.batch_norm_update_stats(input_tensor, running_mean, running_var, momentum)
        running_mean = running_mean.squeeze(0)
        running_var = running_var.squeeze(0)

    if not cpu:
        running_mean = running_mean.cpu()
        running_var = running_var.cpu()

    return {"running_mean": running_mean.numpy().flatten(), "running_var": running_var.numpy().flatten()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"][None, :], dtype=tf.float32)
    running_mean = tf.Variable(input_dict["running_mean"], dtype=tf.float32)
    running_var = tf.Variable(input_dict["running_var"], dtype=tf.float32)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        device_string = "/GPU:0"
    else:
        device_string = "/CPU:0"

    with tf.device(device_string):
        mean_update = running_mean.assign(running_mean * (1 - momentum) + tf.reduce_mean(input_tensor) * momentum)
        variance_update = running_var.assign(running_var * (1 - momentum) + tf.math.reduce_variance(input_tensor) * momentum * (input_tensor.shape[1]/(input_tensor.shape[1]-1)))

        with tf.control_dependencies([mean_update, variance_update]):
            running_mean_np = running_mean.numpy()
            running_var_np = running_var.numpy()
    
    return {"running_mean": running_mean_np.flatten(), "running_var": running_var_np.flatten()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "running_mean": np.array(0.0, dtype=np.float32),
        "running_var": np.array(1.0, dtype=np.float32),
        "momentum": 0.1,
        "eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["running_mean"], tf_result["running_mean"], atol=A_TOL), "running_mean mismatch"
    assert np.allclose(torch_result["running_var"], tf_result["running_var"], atol=A_TOL), "running_var mismatch"

    print("Success")

if __name__ == "__main__":
    main()