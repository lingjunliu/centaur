import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["grad_out"])
    gamma = torch.tensor(input_dict["gamma"])
    mean = torch.tensor(input_dict["mean"])
    invstd = torch.tensor(input_dict["invstd"])
    input_x = torch.tensor(input_dict["input"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    sum_dy = torch.tensor(input_dict["sum_dy"])
    sum_dy_xmu = torch.tensor(input_dict["sum_dy_xmu"])
    count = input_dict.get("count", input_tensor.size()[0])
    if isinstance(count, np.ndarray):
        count = torch.tensor(count)
    else:
        count = torch.tensor(np.array([count], dtype=np.float32))

    if not cpu:
        input_tensor = input_tensor.cuda()
        gamma = gamma.cuda()
        mean = mean.cuda()
        invstd = invstd.cuda()
        input_x = input_x.cuda()
        if weight is not None:
            weight = weight.cuda()
        sum_dy = sum_dy.cuda()
        sum_dy_xmu = sum_dy_xmu.cuda()
        count = count.cuda()
    else:
        input_tensor = input_tensor.cpu()
        gamma = gamma.cpu()
        mean = mean.cpu()
        invstd = invstd.cpu()
        input_x = input_x.cpu()
        if weight is not None:
            weight = weight.cpu()
        sum_dy = sum_dy.cpu()
        sum_dy_xmu = sum_dy_xmu.cpu()
        count = count.cpu()

    result = torch.batch_norm_backward_elemt(
        input_tensor,
        gamma,
        mean,
        invstd,
        input_x,
        sum_dy,
        sum_dy_xmu,
        count
    )

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        grad_out = tf.constant(input_dict["grad_out"])
        gamma = tf.constant(input_dict["gamma"])
        mean = tf.constant(input_dict["mean"])
        invstd = tf.constant(input_dict["invstd"])
        input_x = tf.constant(input_dict["input"])
        sum_dy = tf.constant(input_dict["sum_dy"])
        sum_dy_xmu = tf.constant(input_dict["sum_dy_xmu"])
        count = input_dict.get("count", input_x.shape[0])

        result = grad_out * gamma * invstd

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "grad_out": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "gamma": np.array([0.5, 1.0, 1.5], dtype=np.float32),
        "mean": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "invstd": np.array([2.0, 2.5, 3.0], dtype=np.float32),
        "input": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "sum_dy": np.array([0.5, 1.0, 1.5], dtype=np.float32),
        "sum_dy_xmu": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "count": 3
    }

    torch_result = torch_version(input_data, cpu=True)
    tf_result = tensorflow_version(input_data, cpu=True)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()