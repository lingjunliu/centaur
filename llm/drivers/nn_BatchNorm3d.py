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

    m = torch.nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if not cpu:
        m = m.cuda()

    result = m(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        axes = [0, 2, 3, 4]
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

        if track_running_stats:
            running_mean = tf.Variable(tf.zeros([1, num_features, 1, 1, 1], dtype=tf.float32), trainable=False)
            running_var = tf.Variable(tf.ones([1, num_features, 1, 1, 1], dtype=tf.float32), trainable=False)

            mean_update = running_mean.assign(running_mean * (1 - momentum) + mean * momentum)
            variance_update = running_var.assign(running_var * (1 - momentum) + variance * momentum)
            
            with tf.control_dependencies([mean_update, variance_update]):
                outputs = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, eps)
        else:
            outputs = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, eps)


        if affine:
            gamma = tf.Variable(tf.ones([1, num_features, 1, 1, 1], dtype=tf.float32))
            beta = tf.Variable(tf.zeros([1, num_features, 1, 1, 1], dtype=tf.float32))
            outputs = gamma * outputs + beta
        
        result = outputs.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "num_features": 3,
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