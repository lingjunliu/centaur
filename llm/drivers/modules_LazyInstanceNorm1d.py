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

    layer = torch.nn.LazyInstanceNorm1d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    if not cpu:
        layer = layer.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    result = layer(input_tensor).squeeze(0).squeeze(0)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    def instance_norm(x, epsilon=1e-5, affine=True):
        x = tf.expand_dims(x, axis=0)
        mean, var = tf.nn.moments(x, axes=[1], keepdims=True)
        scale = None
        offset = None
        if affine:
            scale = tf.Variable(tf.ones(mean.shape), dtype=tf.float32)
            offset = tf.Variable(tf.zeros(mean.shape), dtype=tf.float32)
        inv = tf.math.rsqrt(var + epsilon)
        normalized = (x - mean) * inv
        if affine:
            result = scale * normalized + offset
        else:
            result = normalized
        return tf.squeeze(result, axis=0)
    
    if cpu:
        with tf.device("/cpu:0"):
            result = instance_norm(input_tensor, epsilon=eps, affine=affine)
    else:
        with tf.device("/gpu:0"):
            result = instance_norm(input_tensor, epsilon=eps, affine=affine)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
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