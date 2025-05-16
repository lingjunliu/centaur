import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    normalized_shape = input["normalized_shape"]
    eps = input.get("eps", 1e-5)
    elementwise_affine = input.get("elementwise_affine", True)
    bias = input.get("bias", True)

    # Initialize LayerNorm
    layer_norm = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    
    # Apply LayerNorm
    if not cpu:
        layer_norm = layer_norm.cuda()
        input_tensor = input_tensor.cuda()
    
    output = layer_norm(input_tensor)
    
    if not cpu:
        output = output.cpu()

    return {"layer_norm_output": output.detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        normalized_shape = input["normalized_shape"]
        eps = input.get("eps", 1e-5)
        scale = input.get("elementwise_affine", True)

        # Initialize LayerNormalization
        layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=eps, center=scale, scale=scale)
        
        # Apply LayerNormalization
        output = layer_norm(input_tensor)

        return {"layer_norm_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(20, 5, 10).astype(np.float32),
        "normalized_shape": [10],  # Axis: -1 equivalent to the last dimension
        "eps": 1e-5,
        "elementwise_affine": True
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result['layer_norm_output']
    tf_output = tf_result['layer_norm_output']

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()