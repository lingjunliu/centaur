from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Dummy implementation of set_seed function
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    if cpu:
        tensors = [torch.tensor(arr) for arr in input["tensors"]]
    else:
        tensors = [torch.tensor(arr).cuda() for arr in input["tensors"]]

    # Apply torch.hstack
    result_tensor = torch.hstack(tensors)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"hstack_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = [tf.constant(arr) for arr in input["tensors"]]

        # Apply tf.concat equivalent to torch.hstack
        if len(input["tensors"][0].shape) == 1:
            # Concatenate along axis 0 for 1-D tensors
            result_tensor = tf.concat(tensors, axis=0)
        else:
            # Concatenate along axis 1 for all other tensors
            result_tensor = tf.concat(tensors, axis=1)

    return {"hstack_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32)
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print equality check
    torch_hstack_result = torch_result["hstack_result"]
    tf_hstack_result = tf_result["hstack_result"]

    if np.array_equal(torch_hstack_result, tf_hstack_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()