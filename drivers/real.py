import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.complex64)

    # Extract real part using torch.real
    real_tensor = torch.real(input_tensor)

    if not cpu:
        real_tensor = real_tensor.cpu()

    return {"real_part": real_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.complex64)

        # Extract real part using TensorFlow real function
        real_tensor = tf.math.real(input_tensor)

        return {"real_part": real_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.3100 + 0.3553j, -0.5445 - 0.7896j, -1.6492 - 0.0633j, -0.0638 - 0.8119j], dtype=np.complex64)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_real_part = torch_result["real_part"]
    tf_real_part = tf_result["real_part"]

    if np.allclose(torch_real_part, tf_real_part):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()