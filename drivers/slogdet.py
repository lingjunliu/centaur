import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

# PyTorch version
def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["A"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply PyTorch function torch.slogdet (same as torch.linalg.slogdet)
    sign, logabsdet = torch.slogdet(input_tensor)

    if not cpu:
        sign = sign.cpu()
        logabsdet = logabsdet.cpu()

    return {"sign": float(sign.item()), "logabsdet": float(logabsdet.item())}

# TensorFlow version
def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["A"])

        # Apply TensorFlow equivalent
        sign, logabsdet = tf.linalg.slogdet(input_tensor)

        return {"sign": float(sign.numpy()), "logabsdet": float(logabsdet.numpy())}

def main():
    # Example input
    input_data = {
        "A": np.array([[1, 2], [3, 4]], dtype=np.float32)  # Example 2x2 matrix
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if results are equal
    assert np.isclose(torch_result["sign"], tf_result["sign"], atol=1e-6) and \
           np.isclose(torch_result["logabsdet"], tf_result["logabsdet"], atol=1e-6), \
           "Results are not equal!"

    if np.isclose(torch_result["sign"], tf_result["sign"], atol=1e-6) and \
       np.isclose(torch_result["logabsdet"], tf_result["logabsdet"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()