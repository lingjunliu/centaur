import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    dtype = input_dict["dtype"]

    if dtype == "float32":
        dtype_torch = torch.float32
    elif dtype == "float64":
        dtype_torch = torch.float64
    elif dtype == "float16":
        dtype_torch = torch.float16
    elif dtype == "bfloat16":
        dtype_torch = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    if not cpu:
        torch.set_default_dtype(dtype_torch)
    else:
        torch.set_default_dtype(dtype_torch)
    
    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    dtype = input_dict["dtype"]

    if dtype == "float32":
        dtype_tf = tf.float32
    elif dtype == "float64":
        dtype_tf = tf.float64
    elif dtype == "float16":
        dtype_tf = tf.float16
    elif dtype == "bfloat16":
        dtype_tf = tf.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    tf.keras.backend.set_floatx(dtype)
    return {}

def main():
    A_TOL = 0.01

    input_data = {
        "dtype": "float32"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print("Success")

if __name__ == "__main__":
    main()