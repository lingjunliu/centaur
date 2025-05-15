import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict.get("num_features")
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", False)
    track_running_stats = input_dict.get("track_running_stats", False)
    
    if num_features is None and affine:
        num_features = input_tensor.shape[1] 

    if num_features is None and not affine:
        instance_norm = torch.nn.InstanceNorm1d(input_tensor.shape[1], eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    else:
        instance_norm = torch.nn.InstanceNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        instance_norm = instance_norm.cuda()

    result = instance_norm(input_tensor)

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
        eps = input_dict.get("eps", 1e-05)
        affine = input_dict.get("affine", False)

        mean = tf.reduce_mean(input_tensor, axis=2, keepdims=True)
        variance = tf.math.reduce_variance(input_tensor, axis=2, keepdims=True)

        normalized_input = (input_tensor - mean) / tf.sqrt(variance + eps)

        if affine:
            num_features = input_tensor.shape[1]
            gamma = tf.Variable(tf.ones([num_features], dtype=tf.float32))
            beta = tf.Variable(tf.zeros([num_features], dtype=tf.float32))

            gamma = tf.reshape(gamma, [1, -1, 1])
            beta = tf.reshape(beta, [1, -1, 1])

            result = gamma * normalized_input + beta
        else:
            result = normalized_input

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "num_features": 3,
        "affine": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()