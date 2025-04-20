import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply torch.trace
    loss = torch.trace(input_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"trace_sum": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent
        loss = tf.linalg.trace(input_tensor)

        return {"trace_sum": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print results comparison
    assert torch_result["trace_sum"] == tf_result["trace_sum"], "Results do not match!"
    print("equal")

if __name__ == "__main__":
    main()