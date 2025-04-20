from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Ensure to have set_seed function properly defined here
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

def torch_version(input_data, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input_data["input"])
    shape = input_data["shape"]

    # Apply torch.reshape function
    reshaped_tensor = torch.reshape(input_tensor, shape)

    if not cpu and torch.cuda.is_available():
        reshaped_tensor = reshaped_tensor.cpu()

    # Return as numpy array
    return reshaped_tensor.numpy()

def tensorflow_version(input_data, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input_data["input"])
        shape = input_data["shape"]

        # Apply TensorFlow reshape function
        reshaped_tensor = tf.reshape(input_tensor, shape)

        # Return as numpy array
        return reshaped_tensor.numpy()

def main():
    # Example input
    input_data = {
        "input": np.array([[0, 1], [2, 3]], dtype=np.float32),
        "shape": (-1,)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result, tf_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()