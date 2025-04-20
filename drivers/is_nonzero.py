import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.is_nonzero
    result = torch.is_nonzero(input_tensor)

    return {"is_nonzero_result": bool(result)}

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

        # Apply to TensorFlow equivalent by checking if the value is non-zero
        result = tf.reduce_any(tf.not_equal(input_tensor, 0))

        return {"is_nonzero_result": bool(result.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array(1.0),  # Single non-zero value
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert to compare the results
    assert torch_result["is_nonzero_result"] == tf_result["is_nonzero_result"], "Results differ between frameworks."

    if torch_result["is_nonzero_result"] == tf_result["is_nonzero_result"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()