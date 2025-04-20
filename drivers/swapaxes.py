from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Placeholder for set_seed function. Implement this function according to your need.
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def torch_version(input_dict, cpu=True):
    # Set seed for reproducibility
    set_seed()

    input_tensor = torch.tensor(input_dict["input"])
    axis0 = input_dict["axis0"]
    axis1 = input_dict["axis1"]

    result = torch.swapaxes(input_tensor, axis0, axis1)
    
    if not cpu:
        result = result.cpu()
    
    return {"swapped_tensor": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        axis0 = input_dict["axis0"]
        axis1 = input_dict["axis1"]

        perm = list(range(len(input_tensor.shape)))
        perm[axis0], perm[axis1] = perm[axis1], perm[axis0]
        result = tf.transpose(input_tensor, perm=perm)

        return {"swapped_tensor": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32),
        "axis0": 0,
        "axis1": 1,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    np.testing.assert_array_almost_equal(
        torch_result["swapped_tensor"], 
        tf_result["swapped_tensor"],
        decimal=5 
    )
    print("equal")

if __name__ == "__main__":
    main()