import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Perform operation
    result = torch.inner(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"inner_result": result.numpy()}

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

        # Perform operation
        result = tf.tensordot(input_tensor, other_tensor, axes=[[-1], [-1]])

        return {"inner_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.2, 0.5, 0.8], [0.6, 0.1, 0.9]], dtype=np.float32),
        "other": np.array([[0.3, 0.7, 0.4], [0.5, 0.2, 0.6]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert for comparison and print result
    if np.allclose(torch_result["inner_result"], tf_result["inner_result"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()