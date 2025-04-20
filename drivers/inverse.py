import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Transform input to PyTorch tensor
    input_tensor = torch.tensor(input["input"])

    # Apply torch.linalg.inv (torch.inverse is an alias)
    result = torch.inverse(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"inverse": result.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Transform input to TensorFlow tensor
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent using matrix inverse
        result = tf.linalg.inv(input_tensor)

        return {"inverse": result.numpy()}

def main():
    # Example input (using a 3x3 matrix that is not singular)
    input_data = {
        "input": np.array([[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["inverse"], tf_result["inverse"], rtol=1e-5, atol=1e-8):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()