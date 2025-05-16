import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Apply to torch.special.gammainc (which is the same as torch.igamma)
    result = torch.special.gammainc(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"igamma_result": result.numpy()}

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
        result = tf.math.igamma(input_tensor, other_tensor)

        return {"igamma_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "other": np.array([[0.4, 0.1, 0.3], [0.8, 0.7, 0.5]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_result_np = torch_result["igamma_result"]
    tf_result_np = tf_result["igamma_result"]

    # Assert equality
    assert np.allclose(torch_result_np, tf_result_np), "Results differ"
    print("equal")

if __name__ == "__main__":
    main()