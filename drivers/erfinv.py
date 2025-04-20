import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply to torch.special.erfinv
    result_tensor = torch.special.erfinv(input_tensor)

    # Ensure result is on CPU if necessary
    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"erfinv_result": result_tensor.numpy()}

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
        
        # Apply to TensorFlow equivalent
        result_tensor = tf.math.erfinv(input_tensor)

        return {"erfinv_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, -0.3, 0.8], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.allclose(torch_result["erfinv_result"], tf_result["erfinv_result"], atol=1e-6), "Results are not equal"
    
    print("equal")

if __name__ == "__main__":
    main()