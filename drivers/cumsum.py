import numpy as np

# Assuming set_seed is properly imported from src.setseed

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    dtype = torch.float32 if input.get("dtype", None) is None else torch.__dict__[input["dtype"]]

    if cpu:
        input_tensor = input_tensor.cpu()

    # Apply to torch.cumsum
    result_tensor = torch.cumsum(input_tensor, dim=dim, dtype=dtype)
    
    if not cpu:
        result_tensor = result_tensor.cpu()
    
    return {"cumsum_result": result_tensor.numpy()}

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

        # Apply to TensorFlow equivalent
        result_tensor = tf.cumsum(input_tensor, axis=dim)
        
        return {"cumsum_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "dtype": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert the results
    if np.array_equal(torch_result["cumsum_result"], tf_result["cumsum_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()