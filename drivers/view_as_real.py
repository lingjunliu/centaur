import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.complex64)

    # Apply to torch.view_as_real
    real_part = torch.view_as_real(input_tensor)

    if not cpu:
        real_part = real_part.cpu()

    return {"view_as_real": real_part.numpy()}

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

        # Equivalent TensorFlow operation
        real_part = tf.stack([tf.math.real(input_tensor), tf.math.imag(input_tensor)], axis=-1)

        return {"view_as_real": real_part.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Ensure comparison is done in a common format (numpy arrays)
    if np.array_equal(torch_result["view_as_real"], tf_result["view_as_real"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()