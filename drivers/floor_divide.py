import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Apply torch function
    result = torch.floor_divide(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"floor_divide": result.numpy()}

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

        # Apply TensorFlow function
        result = tf.math.floordiv(input_tensor, other_tensor)

        return {"floor_divide": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[9.0, 7.0, 15.0], [16.0, 8.0, 25.0]], dtype=np.float32),
        "other": np.array([[2.0, 3.0, 5.0], [4.0, 2.0, 5.0]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure the results are equivalent
    torch_output = torch_result["floor_divide"]
    tf_output = tf_result["floor_divide"]

    if np.array_equal(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()