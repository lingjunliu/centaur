import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    size = tuple(input["size"])
    fill_value = input["fill_value"]
    dtype = torch.float32  # Only handling float32 for simplicity
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")

    # Create tensor filled with fill_value
    tensor = torch.full(size, fill_value, dtype=dtype, device=device)

    if not cpu:
        tensor = tensor.cpu()

    return {"full_tensor": tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        size = tuple(input["size"])
        fill_value = input["fill_value"]
        dtype = tf.float32  # Only handling float32 for simplicity

        # Create tensor filled with fill_value
        tensor = tf.fill(size, fill_value)

    return {"full_tensor": tensor.numpy()}

def main():
    # Example input
    input_data = {
        "size": [2, 3],
        "fill_value": 3.141592
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["full_tensor"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["full_tensor"])

    # Assert that both results are equal
    if np.array_equal(torch_result["full_tensor"], tf_result["full_tensor"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()