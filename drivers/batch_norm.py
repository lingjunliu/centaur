import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    running_mean = torch.tensor(input["running_mean"])
    running_var = torch.tensor(input["running_var"])
    weight = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    bias = torch.tensor(input.get("bias", None)) if input.get("bias", None) is not None else None
    training = input.get("training", False)
    momentum = input.get("momentum", 0.1)
    eps = input.get("eps", 1e-05)

    # Apply to torch.nn.functional.batch_norm
    output = torch.nn.functional.batch_norm(
        input_tensor, running_mean, running_var, weight=weight, 
        bias=bias, training=training, momentum=momentum, eps=eps
    )

    if not cpu:
        output = output.cpu()

    return {"batch_norm_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        running_mean = tf.constant(input["running_mean"])
        running_var = tf.constant(input["running_var"])
        weight = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        bias = tf.constant(input.get("bias", None)) if input.get("bias", None) is not None else None
        training = input.get("training", False)
        momentum = input.get("momentum", 0.1)
        eps = input.get("eps", 1e-05)

        # Apply Batch norm in TensorFlow
        batch_norm_layer = tf.keras.layers.BatchNormalization(
            momentum=momentum, epsilon=eps, center=bias is not None, scale=weight is not None)
        
        output = batch_norm_layer(
            input_tensor, training=training)

        # Simulation of running mean and var update for tf.keras BatchNormalization
        # Note: This is a simplified assumption
        if not training:
            batch_norm_layer.moving_mean.assign(running_mean)
            batch_norm_layer.moving_variance.assign(running_var)

        return {"batch_norm_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),  # Optional: Can be None
        "bias": np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Optional: Can be None
        "training": False,  # Set to True or False as necessary
        "momentum": 0.1,
        "eps": 1e-05
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion check: Compare the results
    torch_output = torch_result["batch_norm_output"]
    tf_output = tf_result["batch_norm_output"]

    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()