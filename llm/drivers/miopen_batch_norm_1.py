import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])

    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    if cpu:
        result = torch.batch_norm(
            input_tensor,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            cudnn_enabled=False
        )
    else:
        result = torch.miopen_batch_norm(
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
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

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
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        def batch_norm_inference(x, mean, variance, scale, offset, variance_epsilon):
            mean = tf.reshape(mean, [1, -1, 1, 1])
            variance = tf.reshape(variance, [1, -1, 1, 1])
            scale = tf.reshape(scale, [1, -1, 1, 1])
            offset = tf.reshape(offset, [1, -1, 1, 1])
            return scale * (x - mean) / tf.sqrt(variance + variance_epsilon) + offset

        if training:
          mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2])
          result = tf.nn.batch_normalization(input_tensor, mean, variance, bias, weight, eps)

          running_mean_new = running_mean * (1 - momentum) + mean * momentum
          running_var_new = running_var * (1 - momentum) + variance * momentum
          
          result = result.numpy()
        else:
          result = batch_norm_inference(input_tensor, running_mean, running_var, weight, bias, variance_epsilon=eps).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()