import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict.get("num_features")
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", False)
    track_running_stats = input_dict.get("track_running_stats", False)

    if num_features is None and affine:
        num_features = input_tensor.shape[1]

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if num_features is not None:
        norm = torch.nn.InstanceNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    else:
         norm = torch.nn.InstanceNorm3d(input_tensor.shape[1], eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


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
        num_features = input_dict.get("num_features")
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", False)
        track_running_stats = input_dict.get("track_running_stats", False)
        
        if num_features is None:
            num_features = input_tensor.shape[1]
        
        axes = [2, 3, 4]
        mean = tf.math.reduce_mean(input_tensor, axis=axes, keepdims=True)
        variance = tf.math.reduce_variance(input_tensor, axis=axes, keepdims=True)
        
        normalized = (input_tensor - mean) / tf.math.sqrt(variance + eps)
        
        if affine:
            gamma = tf.Variable(tf.ones([1, num_features, 1, 1, 1], dtype=tf.float32))
            beta = tf.Variable(tf.zeros([1, num_features, 1, 1, 1], dtype=tf.float32))
            
            result = gamma * normalized + beta
        else:
            result = normalized

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "num_features": 3,
        "affine": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()