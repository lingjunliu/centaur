import numpy as np

# PyTorch GroupNorm implementation
def torch_version_groupnorm(input, cpu=True):
    import torch
    
    num_groups = input["num_groups"]
    num_channels = input["num_channels"]
    eps = input.get("eps", 1e-05)
    affine = input.get("affine", True)
    
    # Create GroupNorm layer
    groupnorm_layer = torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)
    
    # Generate a tensor from input data
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        groupnorm_layer = groupnorm_layer.cuda()
    
    # Apply GroupNorm
    output = groupnorm_layer(input_tensor)
    
    if not cpu:
        output = output.cpu()
        
    return {"groupnorm_output": output.detach().numpy()}

# TensorFlow GroupNormalization equivalent implementation
def tensorflow_version_groupnorm(input, cpu=True):
    import tensorflow as tf
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    num_groups = input["num_groups"]
    num_channels = input["num_channels"]
    eps = input.get("eps", 1e-05)
    affine = input.get("affine", True)

    if num_channels % num_groups != 0:
        raise ValueError("Number of groups must be a divisor of the number of channels for TensorFlow implementation.")

    with tf.device(device_string):
        # Generate a tensor from input data
        input_tensor = tf.constant(input["input"])
        shape = tf.shape(input_tensor)
        N = shape[0]
        H = shape[2]
        W = shape[3]
        C = num_channels

        # Reshape input tensor to (N, num_groups, channels_per_group, H, W)
        reshaped_input = tf.reshape(input_tensor, (N, num_groups, num_channels // num_groups, H, W))
        
        # Calculate mean and variance over the groups
        mean, variance = tf.nn.moments(reshaped_input, axes=[2, 3, 4], keepdims=True)
        
        # Normalize the input
        normalized_input = (reshaped_input - mean) / tf.sqrt(variance + eps)
        normalized_input = tf.reshape(normalized_input, shape)
        
        if affine:
            gamma = tf.ones((1, C, 1, 1))
            beta = tf.zeros((1, C, 1, 1))
            normalized_input = normalized_input * gamma + beta
        
        return {"groupnorm_output": normalized_input.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(20, 6, 10, 10).astype(np.float32),
        "num_groups": 3,
        "num_channels": 6,
        "eps": 1e-05,
        "affine": True
    }

    # Ensure that the number of groups is a valid divisor of the number of channels
    if input_data["num_channels"] % input_data["num_groups"] != 0:
        print("Number of groups must be a divisor of the number of channels.")
        return

    # PyTorch example
    torch_result = torch_version_groupnorm(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version_groupnorm(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using np.allclose
    if np.allclose(torch_result["groupnorm_output"], tf_result["groupnorm_output"], atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()