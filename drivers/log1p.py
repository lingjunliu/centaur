import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Convert input to torch tensor
    input_tensor = torch.tensor(input["input"])

    # Apply torch.log1p
    output_tensor = torch.log1p(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"log1p": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to tf constant
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow's equivalent log1p function
        output_tensor = tf.math.log1p(input_tensor)

        return {"log1p": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-1.0090, -0.9923, 1.0249, -0.5372, 0.2492], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare results
    try:
        np.testing.assert_allclose(torch_result["log1p"], tf_result["log1p"], rtol=1e-5, atol=1e-5)
        print("equal")
    except AssertionError as e:
        print("not equal")
        print(e)

if __name__ == "__main__":
    main()