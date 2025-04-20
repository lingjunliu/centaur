from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    tensor = torch.tensor(input["tensor"])
    split_size_or_sections = input["split_size_or_sections"]
    dim = input["dim"]

    if not cpu:
        tensor = tensor.cuda()

    # Apply torch.split function
    if isinstance(split_size_or_sections, int):
        result = torch.split(tensor, split_size_or_sections, dim=dim)
    else:
        result = torch.split(tensor, list(split_size_or_sections), dim=dim)

    if not cpu:
        result = tuple(r.cpu() for r in result)  # Move all tensors to CPU if necessary

    return {"split_result": [r.numpy().tolist() for r in result]}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensor = tf.constant(input["tensor"])
        split_size_or_sections = input["split_size_or_sections"]
        dim = input["dim"]

        # Apply TensorFlow equivalent function
        if isinstance(split_size_or_sections, int):
            num_splits = int(tensor.shape[dim]) // split_size_or_sections
            num_remaining = int(tensor.shape[dim]) % split_size_or_sections
            result = tf.split(tensor[:num_splits * split_size_or_sections], num_or_size_splits=num_splits, axis=dim)
            
            # Handle remaining elements
            if num_remaining != 0:
                remaining_tensor = tf.slice(tensor, [num_splits * split_size_or_sections if dim == 0 else 0, 
                                                     num_splits * split_size_or_sections if dim == 1 else 0], [-1, -1])
                result.append(remaining_tensor)
        else:
            result = tf.split(tensor, num_or_size_splits=list(split_size_or_sections), axis=dim)

        return {"split_result": [r.numpy().tolist() for r in result]}

def main():
    # Example input
    input_data = {
        "tensor": np.arange(10).reshape(5, 2).astype(np.float32),
        "split_size_or_sections": 2,
        "dim": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_split_result = np.array(torch_result["split_result"], dtype=object)  # dtype=object to handle different shapes
    tf_split_result = np.array(tf_result["split_result"], dtype=object)
    
    if np.array_equal(torch_split_result, tf_split_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()