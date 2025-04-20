import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.erfc (which is an alias for torch.special.erfc)
    result = torch.erfc(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"erfc_result": result.numpy()}

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

        # Apply to TensorFlow equivalent (tf.math.erfc)
        result = tf.math.erfc(input_tensor)

        return {"erfc_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure that the results are compared as numpy arrays
    if np.allclose(torch_result["erfc_result"], tf_result["erfc_result"], atol=1e-6):
        print("equal")
    else:
        print("not equal")
    
    # Alternatively, more strict comparison without tolerance
    assert np.array_equal(torch_result["erfc_result"], tf_result["erfc_result"]), "Results are not equal"

if __name__ == "__main__":
    main()