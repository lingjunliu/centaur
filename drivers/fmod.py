import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Apply to torch.fmod
    result = torch.fmod(input_tensor, other_tensor)

    if cpu:
        result = result.cpu()

    return {"fmod_result": result.numpy()}

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

        # Manual fmod implementation
        quotient = tf.math.divide_no_nan(input_tensor, other_tensor)
        # Truncate the quotient manually
        truncated_quotient = tf.where(tf.greater_equal(quotient, 0), tf.floor(quotient), tf.math.ceil(quotient))
        truncated_quotient = tf.cast(truncated_quotient)
        fmod_result = input_tensor - truncated_quotient * other_tensor

        return {"fmod_result": fmod_result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-3., -2, -1, 1, 2, 3], dtype=np.float32),
        "other": np.array([2], dtype=np.float32)
    }

    print("Running Torch version:")
    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    print("Running TensorFlow version:")
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy
    if np.allclose(torch_result["fmod_result"], tf_result["fmod_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()