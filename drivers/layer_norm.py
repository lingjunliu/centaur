import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    normalized_shape = input["normalized_shape"]
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    bias_tensor = torch.tensor(input.get("bias", None)) if input.get("bias", None) is not None else None
    eps = input.get("eps", 1e-05)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if weight_tensor is not None:
            weight_tensor = weight_tensor.cuda()
        if bias_tensor is not None:
            bias_tensor = bias_tensor.cuda()

    # Apply to torch.nn.functional.layer_norm
    result = torch.nn.functional.layer_norm(
        input_tensor, normalized_shape, weight=weight_tensor, bias=bias_tensor, eps=eps
    )

    if not cpu:
        result = result.cpu()

    return {"layer_norm_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_str = "/cpu:0"
    else:
        device_str = "/gpu:0"

    with tf.device(device_str):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        normalized_shape = input["normalized_shape"]
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        bias_tensor = tf.constant(input.get("bias", None)) if input.get("bias", None) is not None else None
        eps = input.get("eps", 1e-05)

        # Apply to TensorFlow equivalent
        result = tf.keras.layers.LayerNormalization(
            axis=tuple(range(-len(normalized_shape), 0)), epsilon=eps,
            scale=(weight_tensor is not None), center=(bias_tensor is not None)
        )(input_tensor)

        if weight_tensor is not None:
            result = result * weight_tensor

        if bias_tensor is not None:
            result = result + bias_tensor

        return {"layer_norm_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),  # Random input tensor
        "normalized_shape": [4],
        "weight": np.random.rand(4).astype(np.float32),  # Random weight tensor
        "bias": np.random.rand(4).astype(np.float32),  # Random bias tensor
        "eps": 1e-05
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["layer_norm_result"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["layer_norm_result"])

    # Ensure the results are compared in a common format
    assert np.allclose(torch_result["layer_norm_result"], tf_result["layer_norm_result"], atol=1e-5), "Results are not equal"
    print("Results are equal")

if __name__ == "__main__":
    main()