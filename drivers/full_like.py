from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

def torch_full_like_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    fill_value = input["fill_value"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply torch.full_like
    result_tensor = torch.full_like(input_tensor, fill_value)
    
    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"full_like_result": result_tensor.numpy()}

def tensorflow_full_like_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        fill_value = input["fill_value"]

        # Apply TensorFlow equivalent
        result_tensor = tf.fill(tf.shape(input_tensor), fill_value)

        return {"full_like_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "fill_value": 1.5
    }

    # Torch example
    torch_result = torch_full_like_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_full_like_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert results to numpy arrays for comparison
    torch_result_np = np.array(torch_result["full_like_result"])
    tf_result_np = np.array(tf_result["full_like_result"])

    # Check if the results are equal
    if np.array_equal(torch_result_np, tf_result_np):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()