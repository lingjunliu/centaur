import numpy as np

# Ensure reproducibility
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    dtype = torch.float32 if input.get("dtype", None) is None else torch.dtype(input["dtype"])

    # Apply to torch.nn.functional.log_softmax
    log_softmax = torch.nn.functional.log_softmax(input_tensor, dim=dim, dtype=dtype)

    if not cpu:
        log_softmax = log_softmax.cpu()

    return {"log_softmax": log_softmax.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", None)

        # Apply to TensorFlow equivalent
        log_softmax = tf.nn.log_softmax(input_tensor, axis=dim)

        return {"log_softmax": log_softmax.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "dim": 1,
        "dtype": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_output = np.array(torch_result["log_softmax"])
    tf_output = np.array(tf_result["log_softmax"])

    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()