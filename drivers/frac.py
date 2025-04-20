import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Convert input to a PyTorch tensor
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.frac to compute the fractional portion
    fractional_tensor = torch.frac(input_tensor)

    if not cpu:
        fractional_tensor = fractional_tensor.cpu()

    return {"frac_result": fractional_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    device_string = "/cpu:0" if cpu else "/gpu:0"
    with tf.device(device_string):
        # Convert input to a TensorFlow tensor
        input_tensor = tf.constant(input["input"])

        # Apply equivalent function in TensorFlow
        fractional_tensor = input_tensor - tf.math.floor(tf.math.abs(input_tensor)) * tf.math.sign(input_tensor)

        return {"frac_result": fractional_tensor.numpy()}

# Now, for the main function which tests these functions
def main():
    # Example input data
    input_data = {
        "input": np.array([[1, 2.5, -3.2], [4.2, -5.5, 6.7]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print if results are equal
    assert np.allclose(torch_result["frac_result"], tf_result["frac_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()