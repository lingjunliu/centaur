import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])

    training = input_dict.get("training", False)
    exponential_average_factor = input_dict.get("exponential_average_factor", 0.0)
    eps = input_dict.get("eps", 1e-05)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    if cpu:
        num_features = input_tensor.shape[1]
        result = torch.batch_norm(
            input_tensor,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            exponential_average_factor,
            eps,
            cudnn_enabled=False
        )
        new_running_mean = running_mean
        new_running_var = running_var
    else:
        result, new_running_mean, new_running_var = torch.miopen_batch_norm(
            input_tensor,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            exponential_average_factor,
            eps,
        )
    
    if not cpu:
        result = result.cpu()
        new_running_mean = new_running_mean.cpu()
        new_running_var = new_running_var.cpu()
    
    return {
        "result": result.numpy(),
        "new_running_mean": new_running_mean.numpy(),
        "new_running_var": new_running_var.numpy(),
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        
        training = input_dict.get("training", False)
        exponential_average_factor = input_dict.get("exponential_average_factor", 0.0)
        eps = input_dict.get("eps", 1e-05)

        if training:
            axes = list(range(len(input_tensor.shape) - 1))
            mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

            new_running_mean = (
                exponential_average_factor * mean + (1 - exponential_average_factor) * tf.reshape(running_mean, (1,1,1,-1))
            )
            new_running_var = (
                exponential_average_factor * variance + (1 - exponential_average_factor) * tf.reshape(running_var, (1,1,1,-1))
            )
        else:
            mean = running_mean
            variance = running_var
            new_running_mean = running_mean
            new_running_var = running_var

        result = tf.nn.batch_normalization(
            input_tensor, mean, variance, bias, weight, eps
        )

        result = result.numpy()
        new_running_mean = new_running_mean.numpy()
        new_running_var = new_running_var.numpy()
    
    return {
        "result": result,
        "new_running_mean": new_running_mean,
        "new_running_var": new_running_var,
    }

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "running_mean": np.random.rand(4).astype(np.float32),
        "running_var": np.random.rand(4).astype(np.float32),
        "weight": np.random.rand(4).astype(np.float32),
        "bias": np.random.rand(4).astype(np.float32),
        "training": True,
        "exponential_average_factor": 0.1,
        "eps": 1e-05,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["new_running_mean"], tf_result["new_running_mean"], atol=A_TOL), "new_running_mean Results do not match"
    assert np.allclose(torch_result["new_running_var"], tf_result["new_running_var"], atol=A_TOL), "new_running_var Results do not match"
    print("Success")

if __name__ == "__main__":
    main()