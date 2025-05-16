import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.cfloat)

    # Apply PyTorch imag function
    imag_tensor = torch.imag(input_tensor)

    if not cpu:
        imag_tensor = imag_tensor.cpu()

    return {"imaginary_values": imag_tensor.numpy()}

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

        # Apply TensorFlow function to extract imaginary part
        imag_tensor = tf.math.imag(input_tensor)

        return {"imaginary_values": imag_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.3100+0.3553j, -0.5445-0.7896j, -1.6492-0.0633j, -0.0638-0.8119j], dtype=np.complex64)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if the results are equal
    np.testing.assert_allclose(torch_result["imaginary_values"], tf_result["imaginary_values"], rtol=1e-5, atol=1e-8)

    if np.allclose(torch_result["imaginary_values"], tf_result["imaginary_values"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()