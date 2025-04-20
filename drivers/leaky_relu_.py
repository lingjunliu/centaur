from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    negative_slope = input.get("negative_slope", 0.01)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.leaky_relu_
    torch.nn.functional.leaky_relu_(input_tensor, negative_slope=negative_slope)

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"leaky_relu_output": input_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_arr = np.array(input["input"], dtype=np.float32)
        negative_slope = input.get("negative_slope", 0.01)

        # Apply leaky relu directly to numpy array to simulate in-place operation
        input_arr = np.where(input_arr > 0, input_arr, input_arr * negative_slope)

        # Convert to tensor to keep outputs consistent
        input_tensor = tf.constant(input_arr)

        return {"leaky_relu_output": input_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [0.2, -0.6, 0.9]], dtype=np.float32),
        "negative_slope": 0.01
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert outputs to numpy arrays for comparison
    torch_output = torch_result["leaky_relu_output"]
    tf_output = tf_result["leaky_relu_output"]

    if np.array_equal(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()