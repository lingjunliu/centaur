# Import necessary libraries
from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Function to set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function implementing torch.cummin
def torch_cummin(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Extract parameters from input data
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch function
    values, indices = torch.cummin(input_tensor, dim)

    if not cpu:
        values = values.cpu()
        indices = indices.cpu()

    return {"values": values.numpy(), "indices": indices.numpy()}

# Function implementing TensorFlow equivalent of torch.cummin
def tensorflow_cummin(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Extract parameters from input data
        input_tensor = tf.constant(input["input"])
        dim = input["dim"]

        # Create tensors to hold cumulative minimum values and indices
        values = tf.TensorArray(dtype=tf.float32, size=input_tensor.shape[dim])
        indices = tf.TensorArray(dtype=tf.int64, size=input_tensor.shape[dim])
        
        cumulative_min = tf.constant(np.inf)
        cumulative_idx = tf.constant(-1, dtype=tf.int64)
        
        for i in range(input_tensor.shape[dim]):
            current_val = input_tensor[i] if dim == 0 else input_tensor[:, i]
            cumulative_min = tf.reduce_min([cumulative_min, current_val])
            cumulative_idx = tf.where(cumulative_min == current_val, i, cumulative_idx)
            
            values = values.write(i, cumulative_min)
            indices = indices.write(i, cumulative_idx)
        
        values = values.stack()
        indices = indices.stack()

        return {"values": values.numpy(), "indices": indices.numpy()}

# Main function to test the implementations
def main():
    # Define the input data
    input_data = {
        "input": np.random.randn(10).astype(np.float32),
        "dim": 0
    }

    # PyTorch example
    torch_result = torch_cummin(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_cummin(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results for equality
    assert np.allclose(torch_result["values"], tf_result["values"], atol=1e-6), "Values do not match!"
    assert np.array_equal(torch_result["indices"], tf_result["indices"]), "Indices do not match!"
    print("Results: equal")

# Run the main function
if __name__ == "__main__":
    main()