import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    batchnorm = torch.nn.LazyBatchNorm3d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    if not cpu:
        batchnorm = batchnorm.cuda()
        
    result = batchnorm(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        
        num_features = input_tensor.shape[1]
        
        gamma = tf.Variable(tf.ones([num_features,]), trainable=affine, dtype=tf.float32)
        beta = tf.Variable(tf.zeros([num_features,]), trainable=affine, dtype=tf.float32)
        
        moving_mean = tf.Variable(tf.zeros([num_features,]), trainable=False, dtype=tf.float32)
        moving_variance = tf.Variable(tf.ones([num_features,]), trainable=False, dtype=tf.float32)
        
        if not track_running_stats:
            moving_mean = None
            moving_variance = None

        def batch_norm_training(input_tensor, moving_mean, moving_variance, momentum, eps, gamma, beta):
            axes = list(range(len(input_tensor.shape) - 1))
            mean, variance = tf.nn.moments(input_tensor, axes=axes)
            
            if moving_mean is not None and moving_variance is not None:
                moving_mean.assign(moving_mean * momentum + moving_mean * (1 - momentum))
                moving_variance.assign(moving_variance * momentum + moving_variance * (1 - momentum))
                
            y = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)
            return y

        def batch_norm_inference(input_tensor, moving_mean, moving_variance, eps, gamma, beta):
            y = tf.nn.batch_normalization(input_tensor, moving_mean, moving_variance, beta, gamma, eps)
            return y

        if track_running_stats:
            result = batch_norm_training(input_tensor, moving_mean, moving_variance, momentum, eps, gamma, beta)
            axes = list(range(len(input_tensor.shape) - 1))
            mean, variance = tf.nn.moments(input_tensor, axes=axes)
            moving_mean.assign(momentum * moving_mean + (1 - momentum) * mean)
            moving_variance.assign(momentum * moving_variance + (1 - momentum) * variance)
        else:
            axes = list(range(len(input_tensor.shape) - 1))
            mean, variance = tf.nn.moments(input_tensor, axes=axes)
            result = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)
        return result

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
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