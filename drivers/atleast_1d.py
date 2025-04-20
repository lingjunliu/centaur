import numpy as np

# PyTorch version of torch.atleast_1d
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    tensors = input["tensors"]

    # Move tensors to appropriate device if needed
    if not cpu:
        tensors = [torch.tensor(t).cuda() for t in tensors]
    else:
        tensors = [torch.tensor(t) for t in tensors]

    # Use torch.atleast_1d
    result = torch.atleast_1d(*tensors)

    # Switch to CPU if specified
    if not cpu:
        result = [r.cpu() for r in result]

    # Convert the list of torch tensors back to numpy arrays
    result_np = [r.numpy() for r in result]

    return {"atleast_1d_result": result_np}

# TensorFlow version of tf.atleast_1d
def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = input["tensors"]
        
        # Convert inputs to TensorFlow tensors
        tf_tensors = [tf.constant(t) for t in tensors]

        # Use tf.atleast_1d equivalent
        result = [tf.expand_dims(t, 0) if t.shape.ndims == 0 else t for t in tf_tensors]

        # Convert the list of TensorFlow tensors back to numpy arrays
        result_np = [r.numpy() for r in result]

        return {"atleast_1d_result": result_np}

# Main function to test both implementations
def main():
    # Example input
    input_data = {
        "tensors": [np.array(1.0, dtype=np.float32), np.array([2.0, 3.0], dtype=np.float32), np.array([[4.0, 5.0]], dtype=np.float32)]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both results to a common format (numpy arrays) for comparison
    torch_atleast_1d_result = torch_result["atleast_1d_result"]
    tf_atleast_1d_result = tf_result["atleast_1d_result"]

    for torch_res, tf_res in zip(torch_atleast_1d_result, tf_atleast_1d_result):
        assert np.array_equal(torch_res, tf_res), f"Result mismatch: Torch {torch_res} != TensorFlow {tf_res}"

    print("equal")

if __name__ == "__main__":
    main()