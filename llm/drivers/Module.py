import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict.get("weight", np.ones_like(input_dict["running_var"])))
    bias = torch.tensor(input_dict.get("bias", np.zeros_like(input_dict["running_mean"])))
    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.nn.functional.batch_norm(
        input_tensor,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps
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
        input_tensor = tf.constant(input_dict["input"])
        running_mean = tf.Variable(input_dict["running_mean"], trainable=False, dtype=tf.float32)
        running_var = tf.Variable(input_dict["running_var"], trainable=False, dtype=tf.float32)
        weight = tf.constant(input_dict.get("weight", np.ones_like(input_dict["running_var"])), dtype=tf.float32)
        bias = tf.constant(input_dict.get("bias", np.zeros_like(input_dict["running_mean"])), dtype=tf.float32)
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        def training_batch_norm(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
            axes = list(range(len(input_tensor.shape) - 1))
            mean, variance = tf.nn.moments(input_tensor, axes=axes)
            result = tf.nn.batch_normalization(input_tensor, mean, variance, offset=bias, scale=weight, variance_epsilon=eps)
            
            running_mean.assign(running_mean * momentum + mean * (1 - momentum))
            running_var.assign(running_var * momentum + variance * (1 - momentum))
            
            return result

        def inference_batch_norm(input_tensor, running_mean, running_var, weight, bias, eps):
            return tf.nn.batch_normalization(input_tensor, running_mean, running_var, offset=bias, scale=weight, variance_epsilon=eps)
        
        if training:
            result = training_batch_norm(input_tensor, running_mean, running_var, weight, bias, momentum, eps)
        else:
            result = inference_batch_norm(input_tensor, running_mean, running_var, weight, bias, eps)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    num_features = 4
    input_data = {
        "input": np.random.rand(2, 4, 4, num_features).astype(np.float32),
        "running_mean": np.random.rand(num_features).astype(np.float32),
        "running_var": np.random.rand(num_features).astype(np.float32),
        "weight": np.random.rand(num_features).astype(np.float32),
        "bias": np.random.rand(num_features).astype(np.float32),
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