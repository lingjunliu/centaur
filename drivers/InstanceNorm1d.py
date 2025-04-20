import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def pytorch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    num_features = input["num_features"]
    eps = input.get("eps", 1e-05)
    momentum = input.get("momentum", 0.1)
    affine = input.get("affine", False)
    track_running_stats = input.get("track_running_stats", False)

    # Initialize InstanceNorm1d layer with given parameters
    layer = torch.nn.InstanceNorm1d(num_features, eps=eps, momentum=momentum,
                                    affine=affine, track_running_stats=track_running_stats)
    
    input_tensor = torch.tensor(input["input"])

    # Apply layer
    if not cpu:
        input_tensor = input_tensor.cuda()
        layer = layer.cuda()

    output = layer(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"instance_norm_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    num_features = input["num_features"]
    eps = input.get("eps", 1e-05)

    if cpu:
        device_string = "/CPU:0"
    else:
        device_string = "/GPU:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        
        def instance_norm(tensor, num_features, eps=1e-5):
            mean, variance = tf.nn.moments(tensor, axes=[2], keepdims=True)
            normalized = (tensor - mean) / tf.sqrt(variance + eps)
            return normalized

        output = instance_norm(input_tensor, num_features, eps)

    return {"instance_norm_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(20, 100, 40).astype(np.float32),
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False
    }

    # PyTorch example
    torch_result = pytorch_version(input_data)
    print("PyTorch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_output = torch_result["instance_norm_output"]
    tf_output = tf_result["instance_norm_output"]
    
    # Assert that ensures the outputs are the same
    assert np.allclose(torch_output, tf_output, atol=1e-5), "Outputs are not equal"
    print("equal")

if __name__ == "__main__":
    main()