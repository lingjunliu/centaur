import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Convert input to tensor
    input_tensor = torch.tensor(input["input"])

    # Compute the pseudoinverse using PyTorch
    pinverse_tensor = torch.pinverse(input_tensor, rcond=1e-15)

    if not cpu:
        pinverse_tensor = pinverse_tensor.cpu()

    return {"pinverse": pinverse_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Convert input to tensor
        input_tensor = tf.constant(input["input"])

        # Compute the pseudoinverse using TensorFlow
        pinverse_tensor = tf.linalg.pinv(input_tensor, rcond=1e-15)

        return {"pinverse": pinverse_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9], [0.1, 0.4, 0.7]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert the results are equal
    torch_output = torch_result["pinverse"]
    tf_output = tf_result["pinverse"]
    assert np.allclose(torch_output, tf_output, atol=1e-6), "Results are not equal"
    print("Results are equal")

if __name__ == "__main__":
    main()