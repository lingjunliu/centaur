import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.bool)
    other_tensor = torch.tensor(input["other"], dtype=torch.bool)

    # Apply torch.logical_and
    result = torch.logical_and(input_tensor, other_tensor)

    if cpu:
        result = result.cpu()

    return {"logical_and_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.bool)
        other_tensor = tf.constant(input["other"], dtype=tf.bool)

        # Apply TensorFlow equivalent
        result = tf.logical_and(input_tensor, other_tensor)

        return {"logical_and_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
        "other": np.array([[True, True, False], [False, True, True]], dtype=np.bool_)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:\n", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:\n", tf_result)

    # Compare and assert the results
    torch_output = torch_result["logical_and_result"]
    tf_output = tf_result["logical_and_result"]

    if np.array_equal(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()