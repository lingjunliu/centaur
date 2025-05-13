import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    eps = input_dict.get("eps", 1e-5)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    norm = torch.nn.InstanceNorm3d(input_tensor.shape[1], eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    norm = norm.to(input_tensor.device)

    result = norm(input_tensor)

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

        eps = input_dict.get("eps", 1e-5)
        affine = input_dict.get("affine", True)

        num_features = input_tensor.shape[1]
        mean, variance = tf.nn.moments(input_tensor, axes=[2, 3, 4], keepdims=True)

        if affine:
            gamma = tf.Variable(tf.ones([num_features], dtype=tf.float32))
            beta = tf.Variable(tf.zeros([num_features], dtype=tf.float32))

            gamma = tf.reshape(gamma, (1, num_features, 1, 1, 1))
            beta = tf.reshape(beta, (1, num_features, 1, 1, 1))
            
            normalized = (input_tensor - mean) / tf.sqrt(variance + eps)
            result = gamma * normalized + beta
        else:
            result = (input_tensor - mean) / tf.sqrt(variance + eps)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_shape = (2, 3, 4, 5, 6)
    input_data = {
        "input": np.random.rand(*input_shape).astype(np.float32),
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()