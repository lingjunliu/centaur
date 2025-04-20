import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Compute element-wise greater than
    result = torch.gt(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

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

        # Compute element-wise greater than
        result = tf.math.greater(input_tensor, other_tensor)

        return {"result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "other": np.array([[1, 1], [4, 4]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if the results are equal
    assert np.array_equal(torch_result["result"], tf_result["result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()