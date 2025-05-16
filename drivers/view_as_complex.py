import numpy as np

def torch_view_as_complex_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply torch.view_as_complex
    complex_tensor = torch.view_as_complex(input_tensor)

    if not cpu:
        complex_tensor = complex_tensor.cpu()

    return {"view_as_complex": complex_tensor.numpy()}

def tf_view_as_complex_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Convert to complex representation (TensorFlow does not have direct equivalent of view_as_complex)
        complex_tensor = tf.complex(input_tensor[..., 0], input_tensor[..., 1])

        return {"view_as_complex": complex_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_view_as_complex_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tf_view_as_complex_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion and comparison
    torch_output = torch_result["view_as_complex"]
    tf_output = tf_result["view_as_complex"]

    if np.allclose(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()