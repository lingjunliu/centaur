from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

def set_seed(seed=42):
    # Function to set seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Convert input to torch tensor
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Compute nonzero indices
    nonzero_indices = torch.nonzero(input_tensor, as_tuple=input.get("as_tuple", False))

    if not cpu:
        if isinstance(nonzero_indices, tuple):
            nonzero_indices = tuple(tensor.cpu() for tensor in nonzero_indices)
        else:
            nonzero_indices = nonzero_indices.cpu()

    if isinstance(nonzero_indices, tuple):
        return {"nonzero_indices": [tensor.numpy() for tensor in nonzero_indices]}
    return {"nonzero_indices": nonzero_indices.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to TensorFlow tensor
        input_tensor = tf.constant(input["input"])

        # Compute nonzero indices
        nonzero_indices = tf.where(tf.not_equal(input_tensor, 0))

    return {"nonzero_indices": nonzero_indices.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.6, 0.0, 0.0, 0.0],
                           [0.0, 0.4, 0.0, 0.0],
                           [0.0, 0.0, 1.2, 0.0],
                           [0.0, 0.0, 0.0, -0.4]], dtype=np.float32),
        "as_tuple": False  # Change to True to test the tuple case
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_result_np = torch_result["nonzero_indices"]
    tf_result_np = tf_result["nonzero_indices"]

    if np.array_equal(torch_result_np, tf_result_np):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()