import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    dtype = input.get("dtype", None)
    if dtype is not None:
        dtype = getattr(torch, dtype)

    # Apply torch.cumprod
    if dtype is not None:
        loss = torch.cumprod(input_tensor, dim=dim, dtype=dtype)
    else:
        loss = torch.cumprod(input_tensor, dim=dim)
    
    if not cpu:
        loss = loss.cpu()

    return {"cumprod": loss.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input["dim"]
        dtype = input.get("dtype", None)
        if dtype is not None:
            dtype = getattr(tf, dtype)

        # Apply TensorFlow equivalent
        if dtype is not None:
            input_tensor = tf.cast(input_tensor, dtype)
            loss = tf.math.cumprod(input_tensor, axis=dim, exclusive=False)
        else:
            loss = tf.math.cumprod(input_tensor, axis=dim, exclusive=False)
        
        return {"cumprod": loss.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.6001, 0.2069, -0.1919, 0.9792, 0.6727, 1.0062, 0.4126, -0.2129, -0.4206, 0.1968], dtype=np.float32),
        "dim": 0,
        "dtype": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["cumprod"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["cumprod"])

    # Comparison
    np.testing.assert_almost_equal(torch_result["cumprod"], tf_result["cumprod"], decimal=5)
    print("equal")

if __name__ == "__main__":
    main()