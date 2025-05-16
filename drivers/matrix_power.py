import numpy as np


def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    n = input["n"]

    # Apply torch.matrix_power
    result = torch.matrix_power(input_tensor, n)

    if not cpu:
        result = result.cpu()

    return {"matrix_power_result": result.numpy()}


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
        n = input["n"]

        # Apply TensorFlow equivalent using tf.linalg.matmul in a loop
        result = tf.eye(tf.shape(input_tensor)[0])
        for _ in range(n):
            result = tf.linalg.matmul(result, input_tensor)

        return {"matrix_power_result": result.numpy()}


def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3], [0.2, 0.6]], dtype=np.float32),
        "n": 3
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure results are compared in a common format
    assert np.allclose(torch_result["matrix_power_result"], tf_result["matrix_power_result"]), "Results differ"
    print("equal")


if __name__ == "__main__":
    main()