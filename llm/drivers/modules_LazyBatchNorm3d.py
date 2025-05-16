import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    eps = input_dict.get("eps", 1e-5)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    bn = torch.nn.LazyBatchNorm3d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    if not cpu:
        bn = bn.cuda()
    
    result = bn(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    eps = input_dict.get("eps", 1e-5)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = input_tensor.shape
        num_features = input_shape[-1] if len(input_shape) > 0 else 1
        
        gamma = tf.Variable(tf.ones([num_features], dtype=tf.float32) if affine else None, trainable=affine)
        beta = tf.Variable(tf.zeros([num_features], dtype=tf.float32) if affine else None, trainable=affine)
        moving_mean = tf.Variable(tf.zeros([num_features], dtype=tf.float32), trainable=False)
        moving_variance = tf.Variable(tf.ones([num_features], dtype=tf.float32), trainable=False)

        axes = list(range(len(input_shape) - 1)) if len(input_shape) > 1 else []

        def training_fn():
            mean, variance = tf.nn.moments(input_tensor, axes=axes)

            update_moving_mean = moving_mean.assign(moving_mean * (1 - momentum) + mean * momentum)
            update_moving_variance = moving_variance.assign(moving_variance * (1 - momentum) + variance * momentum)

            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.nn.batch_normalization(input_tensor, mean, variance, beta if affine else None, gamma if affine else None, eps)

        def inference_fn():
            return tf.nn.batch_normalization(input_tensor, moving_mean, moving_variance, beta if affine else None, gamma if affine else None, eps)

        if track_running_stats:
            result = training_fn()
        else:
            mean, variance = tf.nn.moments(input_tensor, axes=axes)
            result = tf.nn.batch_normalization(input_tensor, mean, variance, beta if affine else None, gamma if affine else None, eps)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()