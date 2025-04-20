import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Apply torch.remainder
    result = torch.remainder(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"remainder_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])

        # Apply TensorFlow equivalent
        result = tf.math.floormod(input_tensor, other_tensor)

        return {"remainder_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "other": np.array([2.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print equality
    assert np.allclose(torch_result["remainder_result"], tf_result["remainder_result"]), "Results are not equal!"
    print("equal")

if __name__ == "__main__":
    main()