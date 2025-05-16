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

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    with torch.no_grad():
        running_mean.copy_(running_mean * momentum + input_tensor.mean() * (1 - momentum))
        running_var.copy_(running_var * momentum + input_tensor.var(unbiased=False) * (1 - momentum))

    if not cpu:
        running_mean = running_mean.cpu()
        running_var = running_var.cpu()

    return {"running_mean": running_mean.numpy(), "running_var": running_var.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        running_mean = tf.Variable(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.Variable(input_dict["running_var"], dtype=tf.float32)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)
        
        mean, variance = tf.nn.moments(input_tensor, axes=[0])
        
        new_running_mean = running_mean.assign(running_mean * momentum + mean * (1 - momentum))
        new_running_var = running_var.assign(running_var * momentum + variance * (1 - momentum))

    return {"running_mean": new_running_mean.numpy(), "running_var": new_running_var.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "running_mean": np.array(0.0, dtype=np.float32),
        "running_var": np.array(1.0, dtype=np.float32),
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