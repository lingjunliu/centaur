import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_batch_norm_2d(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]).to(device)
    num_features = input["num_features"]
    eps = input.get("eps", 1e-05)
    momentum = input.get("momentum", 0.1)
    affine = input.get("affine", True)
    track_running_stats = input.get("track_running_stats", True)

    # Apply PyTorch BatchNorm2d
    model = torch.nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum,
                                 affine=affine, track_running_stats=track_running_stats).to(device)
    output = model(input_tensor)

    output = output.cpu() if not cpu else output

    return {"batch_norm_2d_output": output.detach().numpy()}

def tensorflow_batch_norm_2d(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        num_features = input["num_features"]
        epsilon = input.get("eps", 1e-05)
        momentum = input.get("momentum", 0.1)
        affine = input.get("affine", True)
        track_running_stats = input.get("track_running_stats", True)

        if track_running_stats:
            bn_layer = tf.keras.layers.BatchNormalization(axis=1, epsilon=epsilon, momentum=momentum,
                                                          center=affine, scale=affine)
        else:
            bn_layer = tf.keras.layers.BatchNormalization(axis=1, epsilon=epsilon, momentum=momentum,
                                                          center=affine, scale=affine, moving_mean_initializer=None,
                                                          moving_variance_initializer=None)

        output = bn_layer(input_tensor, training=True) # Ensure to set training as `True` to apply batch statistics

        return {"batch_norm_2d_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(20, 100, 35, 45).astype(np.float32),
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True
    }

    # Torch example
    torch_result = torch_batch_norm_2d(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_batch_norm_2d(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure both outputs are in a common format for comparison (numpy arrays)
    torch_output = torch_result["batch_norm_2d_output"]
    tf_output = tf_result["batch_norm_2d_output"]

    # Assert and print result
    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()