import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"], requires_grad=True)
    eps = input_dict.get("eps", 1e-5)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_bn = torch.nn.LazyBatchNorm3d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if not cpu:
        lazy_bn = lazy_bn.cuda()
    
    result = lazy_bn(input_tensor)

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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-5)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)
        
        input_shape = input_tensor.shape
        num_features = input_shape[-1]

        gamma = tf.Variable(tf.ones(num_features, dtype=tf.float32), trainable=affine, name='gamma')
        beta = tf.Variable(tf.zeros(num_features, dtype=tf.float32), trainable=affine, name='beta')

        mean = tf.Variable(tf.zeros(num_features, dtype=tf.float32), trainable=False, name='mean')
        variance = tf.Variable(tf.ones(num_features, dtype=tf.float32), trainable=False, name='variance')
            
        axes = list(range(len(input_shape) - 1)) if len(input_shape) > 1 else []
        
        def training_fn():
            batch_mean, batch_variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)

            mean_update = mean.assign(mean * (1 - momentum) + batch_mean * momentum)
            variance_update = variance.assign(variance * (1 - momentum) + batch_variance * momentum)
            
            with tf.control_dependencies([mean_update, variance_update]):
                return tf.nn.batch_normalization(input_tensor, batch_mean, batch_variance, beta, gamma, eps)

        def inference_fn():
             return tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)

        def bn():
            if track_running_stats:
                return training_fn()
            else:
                return inference_fn()
        
        result = bn()
        result = result.numpy()

    return {"result": result}

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