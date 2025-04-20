import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    abs_tensor = torch.tensor(input["abs"])
    angle_tensor = torch.tensor(input["angle"])
    
    # Apply torch.polar
    complex_tensor = torch.polar(abs_tensor, angle_tensor)
    
    if cpu:
        complex_tensor = complex_tensor.cpu()

    return {"complex_tensor": complex_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        abs_tensor = tf.constant(input["abs"])
        angle_tensor = tf.constant(input["angle"])

        # Compute real and imaginary parts
        real_part = abs_tensor * tf.cos(angle_tensor)
        imag_part = abs_tensor * tf.sin(angle_tensor)

        # Combine to form a complex tensor
        complex_tensor = tf.complex(real_part, imag_part)

        return {"complex_tensor": complex_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "abs": np.array([1.0, 2.0], dtype=np.float32),
        "angle": np.array([np.pi / 2, 5 * np.pi / 4], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    torch_tensor = torch_result["complex_tensor"]
    tf_tensor = tf_result["complex_tensor"]
    
    # Ensure the comparison is valid by checking for near equality due to floating-point precision
    if np.allclose(torch_tensor, tf_tensor):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()