import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
    
    result = torch.native_batch_norm(
        input_tensor,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps
    )
    
    if not cpu:
        result = (result[0].cpu(), result[1].cpu(), result[2].cpu())
    else:
        result = (result[0], result[1], result[2])
    
    return {"result": result[0].numpy(), "running_mean": result[1].numpy(), "running_var": result[2].numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        def batch_norm(x, mean, variance, offset, scale, epsilon):
            inv = tf.math.rsqrt(variance + epsilon)
            return scale * (x - mean) * inv + offset
        
        if training:
            mean, variance = tf.nn.moments(input_tensor, axes=[0])

            # Update running statistics
            new_running_mean = (1 - momentum) * running_mean + momentum * mean
            new_running_var = (1 - momentum) * running_var + momentum * variance
            
            result = batch_norm(input_tensor, mean, variance, bias, weight, eps)
            
            new_running_mean_np = new_running_mean.numpy()
            new_running_var_np = new_running_var.numpy()
        else:
            result = batch_norm(input_tensor, running_mean, running_var, bias, weight, eps)
            new_running_mean_np = running_mean.numpy()
            new_running_var_np = running_var.numpy()

    return {"result": result.numpy(), "running_mean": new_running_mean_np, "running_var": new_running_var_np}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "weight": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "training": True,
        "momentum": 0.1,
        "eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["running_mean"], tf_result["running_mean"], atol=A_TOL), "Running mean results do not match"
    assert np.allclose(torch_result["running_var"], tf_result["running_var"], atol=A_TOL), "Running var results do not match"

    print("Success")

if __name__ == "__main__":
    main()