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

    if cpu:
        mean = torch.mean(input_tensor)
        var = torch.var(input_tensor, unbiased=False)
    else:
        mean = torch.mean(input_tensor)
        var = torch.var(input_tensor, unbiased=False)


    if not cpu:
        mean = mean.cpu()
        var = var.cpu()
    
    return {"mean": mean.numpy(), "var": var.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        mean, variance = tf.nn.moments(input_tensor, axes=[0])
        
        return {"mean": mean.numpy(), "var": variance.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "momentum": 0.1,
        "eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["mean"], tf_result["mean"], atol=A_TOL), "Mean results do not match"
    assert np.allclose(torch_result["var"], tf_result["var"], atol=A_TOL), "Variance results do not match"

    print("Success")

if __name__ == "__main__":
    main()