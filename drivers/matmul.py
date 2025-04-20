import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    # Apply torch.matmul
    result = torch.matmul(input_tensor, other_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"matmul": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])

        # Apply tf.matmul
        result = tf.matmul(input_tensor, other_tensor)

        return {"matmul": result.numpy()}

def main():
    # Example input for matrix-matrix multiplication
    input_data = {
        "input": np.random.randn(3, 4).astype(np.float32),
        "other": np.random.randn(4, 5).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.allclose(torch_result, tf_result, atol=1e-6), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()