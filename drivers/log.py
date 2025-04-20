import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Compute natural logarithm in PyTorch
    result = torch.log(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"log_result": result.numpy()}

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

        # Compute natural logarithm in TensorFlow
        result = tf.math.log(input_tensor)

    return {"log_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([4.7767, 4.3234, 1.2156, 0.2411, 4.5739], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_log = torch_result["log_result"]
    tf_log = tf_result["log_result"]

    if np.allclose(torch_log, tf_log, atol=1e-6):
        print("equal")
    else:
        print("not equal")

    # Assert to show comparison
    assert np.allclose(torch_log, tf_log, atol=1e-6), "The results are not equal!"

if __name__ == "__main__":
    main()