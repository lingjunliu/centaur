import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    instance_norm = torch.nn.LazyInstanceNorm1d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if not cpu:
        instance_norm = instance_norm.cuda()

    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(1)
    elif len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(1)
    result = instance_norm(input_tensor)

    if not cpu:
        result = result.cpu()

    if len(input_tensor.shape) == 2:
        return {"result": result.detach().squeeze().numpy()}
    else:
        return {"result": result.detach().squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", False)
        momentum = input_dict.get("momentum", 0.1)

        if len(input_tensor.shape) == 1:
            input_tensor = tf.reshape(input_tensor, (1, 1, -1))
        elif len(input_tensor.shape) == 2:
            input_tensor = tf.reshape(input_tensor, (tf.shape(input_tensor)[0], 1, tf.shape(input_tensor)[1]))

        mean, variance = tf.nn.moments(input_tensor, axes=[2], keepdims=True)
        normalized = (input_tensor - mean) / tf.sqrt(variance + eps)

        if affine:
            gamma = tf.Variable(tf.ones([1, 1, tf.shape(input_tensor)[2]], dtype=tf.float32), trainable=True)
            beta = tf.Variable(tf.zeros([1, 1, tf.shape(input_tensor)[2]], dtype=tf.float32), trainable=True)
            result = gamma * normalized + beta
        else:
            result = normalized

        if len(input_tensor.shape) == 3:
            result = tf.reshape(result, (tf.shape(result)[0], tf.shape(result)[2]))
        elif len(input_tensor.shape) == 1:
            result = tf.reshape(result, (-1,))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.01, 1, 1.3, -0.6]], dtype=np.float32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()