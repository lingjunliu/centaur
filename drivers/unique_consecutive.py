import numpy as np
from itertools import groupby

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)

    # Apply to torch.unique_consecutive
    unique_tensor = torch.unique_consecutive(input_tensor, return_inverse=False, return_counts=False, dim=dim)

    if not cpu:
        unique_tensor = unique_tensor.cpu()

    return {"unique_consecutive": unique_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", None)

        # Convert tensor to numpy for easier manipulation
        input_np = input_tensor.numpy()
        
        if dim is None:
            # Flatten the input for simplicity
            input_np = input_np.flatten()

        # Use groupby to replicate the behavior.
        unique_np = np.array([key for key, group in groupby(input_np)])

        # Adjust shape back if a dimension was specified
        if dim is not None:
            unique_np = unique_np.reshape((-1,) + input_np.shape[1:])

    return {"unique_consecutive": unique_np}

def main():
    # Example input
    input_data = {
        "input": np.array([1, 2, 2, 3, 2, 2, 4, 4, 5], dtype=np.float32),
        "dim": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["unique_consecutive"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["unique_consecutive"])

    if np.array_equal(torch_result["unique_consecutive"], tf_result["unique_consecutive"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()