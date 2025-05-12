import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    grad_output = torch.tensor(input_dict["grad_output"])
    mean = torch.tensor(input_dict["mean"])
    invstd = torch.tensor(input_dict["invstd"])
    weight = torch.tensor(input_dict["weight"])
    sum_dy = input_dict.get("sum_dy", False)
    sum_dxmu = input_dict.get("sum_dxmu", False)
    count = input_dict.get("count", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        grad_output = grad_output.cuda()
        mean = mean.cuda()
        invstd = invstd.cuda()
        weight = weight.cuda()

    if cpu:
        result = torch.batch_norm_backward_reduce(grad_output, input_tensor, mean, invstd, weight, sum_dy, sum_dxmu, bool(count))
    else:
        result = torch.batch_norm_backward_reduce(grad_output, input_tensor, mean, invstd, weight, sum_dy, sum_dxmu, bool(count))

    if not cpu:
        result = tuple(r.cpu() for r in result)
    
    return {
        "sum_dy": result[0].numpy(),
        "sum_dxmu": result[1].numpy(),
        "mean_dy": result[2].numpy(),
        "mean_dxmu": result[3].numpy()
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        grad_output = tf.constant(input_dict["grad_output"])
        mean = tf.constant(input_dict["mean"])
        invstd = tf.constant(input_dict["invstd"])
        weight = tf.constant(input_dict["weight"])
        sum_dy = input_dict.get("sum_dy", False)
        sum_dxmu = input_dict.get("sum_dxmu", False)
        count = input_dict.get("count", 0)

        sum_dy_tf = tf.reduce_sum(grad_output, axis=0)
        dxmu = (input_tensor - mean) * invstd
        sum_dxmu_tf = tf.reduce_sum(grad_output * dxmu, axis=0)
        
        if count == 0:
            count_val = tf.cast(tf.shape(input_tensor)[0], dtype=tf.float32)
        else:
            count_val = tf.cast(count, dtype=tf.float32)
        
        mean_dy_tf = sum_dy_tf / count_val
        mean_dxmu_tf = sum_dxmu_tf / count_val

        sum_dy_np = sum_dy_tf.numpy()
        sum_dxmu_np = sum_dxmu_tf.numpy()
        mean_dy_np = mean_dy_tf.numpy()
        mean_dxmu_np = mean_dxmu_tf.numpy()

    return {
        "sum_dy": sum_dy_np,
        "sum_dxmu": sum_dxmu_np,
        "mean_dy": mean_dy_np,
        "mean_dxmu": mean_dxmu_np
    }

def main():
    import torch
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "grad_output": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "mean": np.array([2.0, 3.0], dtype=np.float32),
        "invstd": np.array([0.5, 0.25], dtype=np.float32),
        "weight": np.array([1.0, 1.0], dtype=np.float32),
        "count": 2
    }

    if torch.cuda.is_available():
        torch_result = torch_version(input_data, cpu=False)
    else:
        torch_result = torch_version(input_data, cpu=True)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["sum_dy"], tf_result["sum_dy"], atol=A_TOL), "sum_dy results do not match"
    assert np.allclose(torch_result["sum_dxmu"], tf_result["sum_dxmu"], atol=A_TOL), "sum_dxmu results do not match"
    assert np.allclose(torch_result["mean_dy"], tf_result["mean_dy"], atol=A_TOL), "mean_dy results do not match"
    assert np.allclose(torch_result["mean_dxmu"], tf_result["mean_dxmu"], atol=A_TOL), "mean_dxmu results do not match"

    print("Success")

if __name__ == "__main__":
    main()