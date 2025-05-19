import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    sync_batch_norm = torch.nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats)
    result = sync_batch_norm(input_tensor)
    
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
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)

        gamma_initializer = tf.ones_initializer()
        beta_initializer = tf.zeros_initializer()

        gamma = tf.Variable(gamma_initializer(shape=[num_features]), name='gamma', trainable=affine) if affine else None
        beta = tf.Variable(beta_initializer(shape=[num_features]), name='beta', trainable=affine) if affine else None
        
        running_mean = tf.Variable(tf.zeros([num_features]), trainable=False) if track_running_stats else None
        running_var = tf.Variable(tf.ones([num_features]), trainable=False) if track_running_stats else None

        def batch_norm(x, mean, variance, beta, gamma, epsilon):
            x_normed = (x - mean) / tf.sqrt(variance + epsilon)
            if affine:
                return gamma * x_normed + beta
            else:
                return x_normed

        axes = list(range(len(input_tensor.shape) - 1))
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)

        def update_running_stats():
          m = momentum
          running_mean.assign(m * running_mean + (1 - m) * mean)
          running_var.assign(m * running_var + (1 - m) * variance)
          return running_mean, running_var

        if track_running_stats:
          mean, variance = tf.cond(tf.constant(True), # Replace with training check if needed
                                  lambda: update_running_stats(),
                                  lambda: (running_mean, running_var))
        
        result = batch_norm(input_tensor, mean, variance, beta, gamma, eps)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
        "num_features": 2,
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