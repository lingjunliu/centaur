import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed  # Make sure your set_seed function is correctly implemented

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    start_tensor = torch.tensor(input["start"])
    end_tensor = torch.tensor(input["end"])
    weight_tensor = torch.tensor(input["weight"]) if isinstance(input["weight"], list) else input["weight"]

    # Apply torch.lerp
    result = torch.lerp(start_tensor, end_tensor, weight_tensor)

    if not cpu:
        result = result.cpu()

    return {"lerp_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        start_tensor = tf.constant(input["start"])
        end_tensor = tf.constant(input["end"])
        weight_tensor = tf.constant(input["weight"]) if isinstance(input["weight"], list) else input["weight"]

        # Apply TensorFlow equivalent of torch.lerp
        result = start_tensor + weight_tensor * (end_tensor - start_tensor)

        return {"lerp_result": result.numpy()}

def main():
    # Example input data
    input_data = {
        "start": [1.0, 2.0, 3.0, 4.0],
        "end": [10.0, 10.0, 10.0, 10.0],
        "weight": 0.5  # Can also be a tensor of similar shape
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and comparison
    torch_np = np.array(torch_result["lerp_result"])
    tf_np = np.array(tf_result["lerp_result"])

    assert np.allclose(torch_np, tf_np), "Results are not equal"
    print("equal" if np.allclose(torch_np, tf_np) else "not equal")

if __name__ == "__main__":
    main()