import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    shape = tuple(input["size"])
    dtype = torch.float32 if input.get("dtype", None) is None else input["dtype"]
    device = torch.device("cpu") if cpu else torch.device("cuda")

    tensor = torch.zeros(shape, dtype=dtype, device=device)

    return {"tensor": tensor.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    shape = tuple(input["size"])
    dtype = tf.float32 if input.get("dtype", None) is None else input["dtype"]

    if cpu:
        device = "/cpu:0"
    else:
        device = "/gpu:0"

    with tf.device(device):
        tensor = tf.zeros(shape, dtype=dtype)
        return {"tensor": tensor.numpy()}

def main():
    # Example input
    input_data = {
        "size": [2, 3],  # Shape of the tensor
        "dtype": None  # Using default dtype (float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.array_equal(torch_result["tensor"], tf_result["tensor"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()