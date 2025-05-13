import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    normalized_shape = input_dict["normalized_shape"]
    weight = torch.tensor(input_dict.get("weight", np.ones(normalized_shape, dtype=np.float32)))
    bias = torch.tensor(input_dict.get("bias", np.zeros(normalized_shape, dtype=np.float32)))
    eps = input_dict.get("eps", 1e-5)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, eps)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    normalized_shape = input_dict["normalized_shape"]
    weight = tf.constant(input_dict.get("weight", np.ones(normalized_shape, dtype=np.float32)), dtype=tf.float32)
    bias = tf.constant(input_dict.get("bias", np.zeros(normalized_shape, dtype=np.float32)), dtype=tf.float32)
    eps = input_dict.get("eps", 1e-5)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        mean = tf.math.reduce_mean(input_tensor, axis=[-1], keepdims=True)
        variance = tf.math.reduce_variance(input_tensor, axis=[-1], keepdims=True)
        normalized = (input_tensor - mean) / tf.math.sqrt(variance + eps)
        result = weight * normalized + bias

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "normalized_shape": [3],
        "weight": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "bias": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eps": 1e-5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()