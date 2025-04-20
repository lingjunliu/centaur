import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Compute hyperbolic cosine using PyTorch
    result_tensor = torch.cosh(input_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"cosh_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Compute hyperbolic cosine using TensorFlow
        result_tensor = tf.math.cosh(input_tensor)

        return {"cosh_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.allclose(torch_result["cosh_result"], tf_result["cosh_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()