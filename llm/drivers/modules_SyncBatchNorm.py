import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    sync_batchnorm = torch.nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats)

    result = sync_batchnorm(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        shape = input_tensor.shape
        if len(shape) == 2:
            axes = [0]
            gamma_shape = [num_features]
            beta_shape = [num_features]
        else:
            axes = [0, 1, 2]
            gamma_shape = [num_features]
            beta_shape = [num_features]
            input_tensor = tf.transpose(input_tensor, perm=[0, 3, 1, 2])

        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)

        if track_running_stats:
            running_mean = tf.Variable(tf.zeros([num_features], dtype=tf.float32), trainable=False)
            running_var = tf.Variable(tf.ones([num_features], dtype=tf.float32), trainable=False)
            
            running_mean.assign(running_mean * (1 - momentum) + mean * momentum)
            running_var.assign(running_var * (1 - momentum) + variance * momentum)
            
            mean = running_mean
            variance = running_var

        if affine:
            gamma = tf.Variable(tf.ones(gamma_shape, dtype=tf.float32))
            beta = tf.Variable(tf.zeros(beta_shape, dtype=tf.float32))

            result = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)
        else:
            result = tf.nn.batch_normalization(input_tensor, mean, variance, offset=None, scale=None, variance_epsilon=eps)
            
        if len(shape) > 2:
            result = tf.transpose(result, perm=[0, 2, 3, 1])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 3).astype(np.float32),
        "num_features": 3,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(2, 5).astype(np.float32),
        "num_features": 5,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()