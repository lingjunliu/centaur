import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other = input["other"]

    if isinstance(other, (int, float)):
        other_tensor = torch.tensor(other)
    else:
        other_tensor = torch.tensor(other)

    # Apply torch.lt function
    result = torch.lt(input_tensor, other_tensor).to(torch.bool)

    if not cpu:
        result = result.cpu()

    return {"lt_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other = input["other"]

        if isinstance(other, (int, float)):
            other_tensor = tf.constant(other)
        else:
            other_tensor = tf.constant(other)

        # Apply TensorFlow equivalent
        result = tf.less(input_tensor, other_tensor)

        return {"lt_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "other": np.array([[1, 1], [4, 4]], dtype=np.float32)  # Example other tensor (can also be a scalar)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert the results to numpy arrays for comparison
    torch_lt_result = torch_result["lt_result"]
    tensorflow_lt_result = tf_result["lt_result"]

    if np.array_equal(torch_lt_result, tensorflow_lt_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()