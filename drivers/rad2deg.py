import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.rad2deg
    output_tensor = torch.rad2deg(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"rad2deg": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Convert radians to degrees
        output_tensor = tf.math.multiply(input_tensor, 180.0 / np.pi)

    return {"rad2deg": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing the results
    torch_array = torch_result["rad2deg"]
    tf_array = tf_result["rad2deg"]

    if np.allclose(torch_array, tf_array, rtol=1e-5, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()