import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    norm = torch.nn.LazyInstanceNorm3d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if not cpu:
        norm = norm.cuda()

    result = norm(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

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
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        shape = input_tensor.shape
        axes = list(range(2, len(shape)) if len(shape) > 2 else [])
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

        if affine:
            gamma = tf.Variable(tf.ones(mean.shape), dtype=tf.float32)
            beta = tf.Variable(tf.zeros(mean.shape), dtype=tf.float32)
            result = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)

        else:
            result = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, eps)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 4, 5, 6).astype(np.float32),
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