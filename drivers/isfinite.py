import numpy as np

def torch_version(input, cpu=True):
    import torch

    input_tensor = torch.tensor(input["input"])

    # Compute isfinite using PyTorch
    result = torch.isfinite(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"isfinite_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])

        # Compute isfinite using TensorFlow
        result = tf.math.is_finite(input_tensor)

        return {"isfinite_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, float('inf'), 2.0, float('-inf'), float('nan')], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.array_equal(torch_result["isfinite_result"], tf_result["isfinite_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()