import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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
    
    result = torch.nn.functional.batch_norm(
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
    )

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        num_channels = input_tensor.shape[-1]

        if training:
            mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2], keepdims=False)
            result = tf.nn.batch_normalization(input_tensor, mean, variance, offset=bias, scale=weight, variance_epsilon=eps)

            new_running_mean = running_mean * momentum + mean * (1 - momentum)
            new_running_var = running_var * momentum + variance * (1 - momentum)

            # Not updating the actual running means in this example as the problem description does not require it.
            # In real training, we would update the running means using assign ops.


        else:
            result = tf.nn.batch_normalization(input_tensor, running_mean, running_var, offset=bias, scale=weight, variance_epsilon=eps)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 4, 4, 4).astype(np.float32),
        "weight": np.random.rand(4).astype(np.float32),
        "bias": np.random.rand(4).astype(np.float32),
        "running_mean": np.random.rand(4).astype(np.float32),
        "running_var": np.random.rand(4).astype(np.float32),
        "training": False,
    }


    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()