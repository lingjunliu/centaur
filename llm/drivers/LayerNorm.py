import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    normalized_shape = input_dict["normalized_shape"]
    weight = input_dict.get("weight", None)
    bias = input_dict.get("bias", None)
    eps = input_dict.get("eps", 1e-5)
    elementwise_affine = input_dict.get("elementwise_affine", True)

    if weight is not None:
        weight = torch.tensor(weight)
    if bias is not None:
        bias = torch.tensor(bias)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    layer_norm = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    if weight is not None:
        layer_norm.weight = torch.nn.Parameter(weight)
    else:
        layer_norm.weight = None
    if bias is not None:
        layer_norm.bias = torch.nn.Parameter(bias)
    else:
        layer_norm.bias = None

    result = layer_norm(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    normalized_shape = input_dict["normalized_shape"]
    weight = input_dict.get("weight", None)
    bias = input_dict.get("bias", None)
    eps = input_dict.get("eps", 1e-5)
    elementwise_affine = input_dict.get("elementwise_affine", True)

    if weight is not None:
        weight = tf.constant(weight)
    else:
        weight = tf.ones(normalized_shape)
    if bias is not None:
        bias = tf.constant(bias)
    else:
        bias = tf.zeros(normalized_shape)
    
    axis = list(range(-len(normalized_shape), 0))

    mean = tf.math.reduce_mean(input_tensor, axis=axis, keepdims=True)
    variance = tf.math.reduce_variance(input_tensor, axis=axis, keepdims=True)

    normalized_input = (input_tensor - mean) / tf.math.sqrt(variance + eps)

    if elementwise_affine:
        result = weight * normalized_input + bias
    else:
        result = normalized_input

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "normalized_shape": [3],
        "weight": np.array([0.5, 0.6, 0.7], dtype=np.float32),
        "bias": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eps": 1e-5,
        "elementwise_affine": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "normalized_shape": [3],
        "eps": 1e-5,
        "elementwise_affine": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()