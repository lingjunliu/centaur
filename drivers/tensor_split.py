import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    indices_or_sections = input["indices_or_sections"]
    dim = input.get("dim", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.tensor_split
    result = torch.tensor_split(input_tensor, indices_or_sections, dim)

    if not cpu:
        result = [res.cpu() for res in result]

    return {"tensor_split_result": [res.numpy() for res in result]}

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
        indices_or_sections = input["indices_or_sections"]
        dim = input.get("dim", 0)

        # Apply to TensorFlow equivalent (tf.split for integer indices_or_sections)
        if isinstance(indices_or_sections, int):
            result = tf.split(input_tensor, indices_or_sections, axis=dim)
        else:
            # For indices_or_sections as a list of indices
            result = tf.split(input_tensor, num_or_size_splits=indices_or_sections, axis=dim)

        return {"tensor_split_result": [res.numpy() for res in result]}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9], [0.4, 0.1, 0.7], [0.6, 0.8, 0.2]], dtype=np.float32),
        "indices_or_sections": 2,  # Example case: Split into 2 sections
        "dim": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_result_flat = [np.array(res) for res in torch_result["tensor_split_result"]]
    tf_result_flat = [np.array(res) for res in tf_result["tensor_split_result"]]

    if all(np.array_equal(t, f) for t, f in zip(torch_result_flat, tf_result_flat)):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()