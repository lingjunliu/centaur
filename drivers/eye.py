import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    n = input["n"]
    m = input.get("m", None)
    dtype = torch.float32  # Default PyTorch `eye` dtype

    # Apply to torch.eye
    if m is None:
        eye_tensor = torch.eye(n, dtype=dtype)
    else:
        eye_tensor = torch.eye(n, m, dtype=dtype)

    if not cpu:
        eye_tensor = eye_tensor.cpu()  # Moving to CPU if not already

    return {"eye": eye_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        n = input["n"]
        m = input.get("m", None)
        if m is None:
            m = n

        # Apply to TensorFlow equivalent
        eye_tensor = tf.eye(n, m)

        return {"eye": eye_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "n": 3,
        "m": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:")
    print(torch_result["eye"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:")
    print(tf_result["eye"])

    # Compare results
    if np.array_equal(torch_result["eye"], tf_result["eye"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()