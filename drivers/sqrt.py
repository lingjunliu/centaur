import numpy as np

def torch_sqrt(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply torch.sqrt function
    result = torch.sqrt(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"sqrt_result": result.numpy()}

def tensorflow_sqrt(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply tf.sqrt function
        result = tf.sqrt(input_tensor)

        return {"sqrt_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([4.0, 16.0, 25.0], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_sqrt(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_sqrt(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print equality
    if np.allclose(torch_result["sqrt_result"], tf_result["sqrt_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()