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

    m = torch.nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if not cpu:
        m = m.cuda()
    
    if "running_mean" in input_dict and track_running_stats:
        m.running_mean = torch.tensor(input_dict["running_mean"])
        if not cpu:
            m.running_mean = m.running_mean.cuda()

    if "running_var" in input_dict and track_running_stats:
        m.running_var = torch.tensor(input_dict["running_var"])
        if not cpu:
            m.running_var = m.running_var.cuda()

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
        input_tensor = tf.constant(input_dict["input"])
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        input_shape = input_tensor.shape
        if len(input_shape) == 2:
            axes = [0]
        elif len(input_shape) == 3:
            axes = [0, 2]
        else:
            raise ValueError("Input tensor must be 2D or 3D")

        if track_running_stats:
            if "running_mean" in input_dict and "running_var" in input_dict:
                running_mean = tf.Variable(input_dict["running_mean"], trainable=False, dtype=tf.float32)
                running_var = tf.Variable(input_dict["running_var"], trainable=False, dtype=tf.float32)
            else:
                running_mean = tf.Variable(tf.zeros([num_features], dtype=tf.float32), trainable=False)
                running_var = tf.Variable(tf.ones([num_features], dtype=tf.float32), trainable=False)

            mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)
            
            def update_running_stats():
                update_mean = running_mean.assign(running_mean * (1 - momentum) + mean * momentum)
                update_var = running_var.assign(running_var * (1 - momentum) + variance * momentum)
                with tf.control_dependencies([update_mean, update_var]):
                    return tf.identity(mean), tf.identity(variance)
            
            mean, variance = update_running_stats()

            scale, offset = None, None
            if affine:
                scale = tf.Variable(tf.ones([num_features], dtype=tf.float32))
                offset = tf.Variable(tf.zeros([num_features], dtype=tf.float32))
            
            normalized = tf.nn.batch_normalization(input_tensor, mean, variance, offset, scale, eps)

        else:
            mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)
            
            scale, offset = None, None
            if affine:
                scale = tf.Variable(tf.ones([num_features], dtype=tf.float32))
                offset = tf.Variable(tf.zeros([num_features], dtype=tf.float32))

            normalized = tf.nn.batch_normalization(input_tensor, mean, variance, offset, scale, eps)
            

    return {"result": normalized.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056],
                           [0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "num_features": 4,
        "affine": True,
        "track_running_stats": True,
        "running_mean": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "running_var": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056],
                           [0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "num_features": 4,
        "affine": False,
        "track_running_stats": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()