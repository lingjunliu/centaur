import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Convert input to PyTorch tensor
    input_tensor = torch.tensor(input["input"])

    # Compute reciprocal using PyTorch function
    result_tensor = torch.reciprocal(input_tensor)

    if cpu:
        result_tensor = result_tensor.cpu()

    return {"reciprocal_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device = "/cpu:0"
    else:
        device = "/gpu:0"

    with tf.device(device):
        # Convert input to TensorFlow tensor
        input_tensor = tf.constant(input["input"])

        # Compute reciprocal using TensorFlow function
        result_tensor = tf.math.reciprocal(input_tensor)

        if not cpu:
            result_tensor = tf.identity(result_tensor)  # Ensure it stays on GPU if chosen

    return {"reciprocal_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-0.4595, -2.1219, -1.4314, 0.7298], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if the results are equal
    assert np.allclose(torch_result["reciprocal_result"], tf_result["reciprocal_result"]), "not equal"
    print("equal")

if __name__ == "__main__":
    main()