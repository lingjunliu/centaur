import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["grad_out"])
    gamma = torch.tensor(input_dict["gamma"])
    mean = torch.tensor(input_dict["mean"])
    invstd = torch.tensor(input_dict["invstd"])
    input_tensor_ = torch.tensor(input_dict["input"])
    sum_dy = torch.tensor(input_dict["sum_dy"])
    sum_dy_xmu = torch.tensor(input_dict["sum_dy_xmu"])
    count = torch.tensor(np.array(input_dict["count"]))


    if not cpu:
        input_tensor = input_tensor.cuda()
        gamma = gamma.cuda()
        mean = mean.cuda()
        invstd = invstd.cuda()
        input_tensor_ = input_tensor_.cuda()
        sum_dy = sum_dy.cuda()
        sum_dy_xmu = sum_dy_xmu.cuda()
        count = count.cuda()


    result = torch.batch_norm_backward_elemt(input_tensor, input_tensor_, gamma, mean, invstd, sum_dy, sum_dy_xmu, count)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        grad_out = tf.constant(input_dict["grad_out"])
        gamma = tf.constant(input_dict["gamma"])
        mean = tf.constant(input_dict["mean"])
        invstd = tf.constant(input_dict["invstd"])
        input_tensor = tf.constant(input_dict["input"])
        sum_dy = tf.constant(input_dict["sum_dy"])
        sum_dy_xmu = tf.constant(input_dict["sum_dy_xmu"])
        count = tf.constant(input_dict["count"], dtype=tf.float32)


        result = grad_out * gamma * invstd

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "grad_out": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "gamma": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "mean": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        "invstd": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "input": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "sum_dy": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        "sum_dy_xmu": np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
        "count": 4
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()