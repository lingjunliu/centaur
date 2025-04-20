import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Convert input to tensor
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Compute arcsine
    result = torch.asin(input_tensor)

    # Ensure result is on CPU for consistency
    if not cpu:
        result = result.cpu()

    return {"asin_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    # Choose the correct device
    device_string = "/cpu:0" if cpu else "/gpu:0"

    # Use the device context manager
    with tf.device(device_string):
        # Convert input to tensor
        input_tensor = tf.convert_to_tensor(input["input"])

        # Compute arcsine
        result = tf.math.asin(input_tensor)

    return {"asin_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-0.5, 0.5, -0.8], [0.1, -0.9, 0.3]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both results to numpy arrays for comparison
    torch_result_np = np.array(torch_result["asin_result"])
    tf_result_np = np.array(tf_result["asin_result"])

    # Assert equality and print corresponding message
    if np.allclose(torch_result_np, tf_result_np, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()