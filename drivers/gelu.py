import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply GELU with approximation as 'none'
    gelu_output = torch.nn.functional.gelu(input_tensor, approximate='none')

    if not cpu:
        gelu_output = gelu_output.cpu()

    return {"gelu_output": gelu_output.numpy()}

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

        # Apply GELU equivalent
        gelu_output = input_tensor * 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))

        return {"gelu_output": gelu_output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [0.2, -0.6, 0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy arrays
    if np.allclose(torch_result["gelu_output"], tf_result["gelu_output"], rtol=1e-05, atol=1e-08):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()