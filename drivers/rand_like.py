from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np
import random

def set_seed(seed=0):
    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

def torch_version(input, cpu=True, seed=0):
    # Set seed for reproducibility
    set_seed(seed)

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dtype = torch.float32 if input.get("dtype", None) is None else input["dtype"]
    layout = torch.strided if input.get("layout", None) is None else input["layout"]
    device = torch.device('cpu' if cpu else 'cuda') if input.get("device", None) is None else input["device"]
    requires_grad = input.get("requires_grad", False)
    memory_format = torch.preserve_format if input.get("memory_format", None) is None else input["memory_format"]

    # Generate random data with NumPy
    np_random_data = np.random.rand(*input_tensor.shape).astype(np.float32)

    # Convert NumPy array to Torch tensor
    result = torch.tensor(np_random_data, dtype=dtype, device=device, requires_grad=requires_grad)
    
    # Make sure to apply the same memory format
    result = result.contiguous(memory_format=memory_format)

    if not cpu:
        result = result.cpu()
    
    return {"rand_like": result.numpy()}

def tensorflow_version(input, cpu=True, seed=0):
    # Set seed for reproducibility
    set_seed(seed)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # TensorFlow dtype conversion
        if input.get("dtype", None) is not None:
            if input["dtype"] == torch.float32:
                dtype = tf.float32
            elif input["dtype"] == torch.float64:
                dtype = tf.float64
            else:
                dtype = tf.float32  # Default dtype
        else:
            dtype = tf.float32

        # Generate random data with NumPy (same seed ensures same random values)
        np_random_data = np.random.rand(*input_tensor.shape).astype(np.float32)

        # Convert NumPy array to TensorFlow tensor
        result = tf.convert_to_tensor(np_random_data, dtype=dtype)

        return {"rand_like": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dtype": torch.float32,
        "layout": torch.strided,
        "device": torch.device('cpu'),
        "requires_grad": False,
        "memory_format": torch.preserve_format
    }

    seed = 0

    # Torch example
    torch_result = torch_version(input_data, seed=seed)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, seed=seed)
    print("TensorFlow result:", tf_result)

    # Using numpy to compare both results
    if np.allclose(torch_result["rand_like"], tf_result["rand_like"], rtol=1e-05, atol=1e-08):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()