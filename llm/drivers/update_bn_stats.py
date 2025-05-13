import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic.qat import update_bn_stats

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    momentum_bn = torch.tensor(input_dict["momentum_bn"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        momentum_bn = momentum_bn.cuda()


    update_bn_stats(input_tensor)
    with torch.no_grad():
        running_mean[:] = (1 - momentum_bn) * running_mean + momentum_bn * torch.mean(input_tensor, dim=0)
        running_var[:] = (1 - momentum_bn) * running_var + momentum_bn * torch.var(input_tensor, dim=0, unbiased=False)


    if not cpu:
        running_mean = running_mean.cpu()
        running_var = running_var.cpu()

    return {"running_mean": running_mean.numpy(), "running_var": running_var.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        momentum_bn = tf.constant(input_dict["momentum_bn"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-5)

        mean = tf.reduce_mean(input_tensor, axis=0)
        variance = tf.math.reduce_variance(input_tensor, axis=0)

        new_running_mean = running_mean * (1 - momentum_bn) + momentum_bn * mean
        new_running_var = running_var * (1 - momentum_bn) + momentum_bn * variance

        new_running_mean = new_running_mean.numpy()
        new_running_var = new_running_var.numpy()

    return {"running_mean": new_running_mean, "running_var": new_running_var}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "momentum_bn": np.array(0.1, dtype=np.float32),
        "running_mean": np.array([0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0], dtype=np.float32),
        "eps": 1e-5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["running_mean"], tf_result["running_mean"], atol=A_TOL), "Running mean results do not match"
    assert np.allclose(torch_result["running_var"], tf_result["running_var"], atol=A_TOL), "Running var results do not match"

    print("Success")

if __name__ == "__main__":
    main()