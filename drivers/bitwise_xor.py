import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Apply to torch.bitwise_xor
    result = torch.bitwise_xor(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"bitwise_xor_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])

        # Apply to TensorFlow equivalent
        result = tf.bitwise.bitwise_xor(input_tensor, other_tensor)

        return {"bitwise_xor_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([5, 3, 10], dtype=np.int32),
        "other": np.array([2, 5, 8], dtype=np.int32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print the result
    np.testing.assert_equal(torch_result["bitwise_xor_result"], tf_result["bitwise_xor_result"])
    print("equal")

if __name__ == "__main__":
    main()