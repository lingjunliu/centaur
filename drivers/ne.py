import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Apply to torch.ne
    result = torch.ne(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"ne_result": result.numpy()}

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

        # Apply to tf.not_equal
        result = tf.not_equal(input_tensor, other_tensor)

    return {"ne_result": result.numpy()}

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

    # Assert and print whether results are equal
    assert np.array_equal(torch_result["ne_result"], tf_result["ne_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()