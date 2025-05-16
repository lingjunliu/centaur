import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    normalized_shape = input_dict["normalized_shape"]
    weight = torch.tensor(input_dict.get("weight", np.ones(input_dict["normalized_shape"])))
    bias = torch.tensor(input_dict.get("bias", np.zeros(input_dict["normalized_shape"])))
    eps = input_dict.get("eps", 1e-05)
    elementwise_affine = input_dict.get("elementwise_affine", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.layer_norm(input_tensor, normalized_shape, weight, bias, eps, elementwise_affine)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        normalized_shape = input_dict["normalized_shape"]
        weight = tf.constant(input_dict.get("weight", np.ones(input_dict["normalized_shape"])), dtype=tf.float32)
        bias = tf.constant(input_dict.get("bias", np.zeros(input_dict["normalized_shape"])), dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        elementwise_affine = input_dict.get("elementwise_affine", True)

        mean = tf.math.reduce_mean(input_tensor, axis=list(range(-len(normalized_shape), 0)), keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(input_tensor - mean), axis=list(range(-len(normalized_shape), 0)), keepdims=True)
        normalized = (input_tensor - mean) / tf.math.sqrt(variance + eps)

        if elementwise_affine:
            result = weight * normalized + bias
        else:
            result = normalized

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "normalized_shape": [3],
        "weight": np.array([0.5, 1.0, 1.5], dtype=np.float32),
        "bias": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eps": 1e-05,
        "elementwise_affine": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()