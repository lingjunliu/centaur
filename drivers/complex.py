import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    real = torch.tensor(input["real"])
    imag = torch.tensor(input["imag"])

    if not cpu:
        real = real.cuda()
        imag = imag.cuda()

    # Apply to torch.complex
    complex_tensor = torch.complex(real, imag)

    if not cpu:
        complex_tensor = complex_tensor.cpu()

    return {"complex_tensor": complex_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        real = tf.constant(input["real"])
        imag = tf.constant(input["imag"])

        # Combine real and imaginary parts to form a complex tensor
        complex_tensor = tf.complex(real, imag)

        return {"complex_tensor": complex_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "real": np.array([1.0, 2.0], dtype=np.float32),
        "imag": np.array([3.0, 4.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["complex_tensor"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["complex_tensor"])

    # Assert that the results are equal
    assert np.array_equal(torch_result["complex_tensor"], tf_result["complex_tensor"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()