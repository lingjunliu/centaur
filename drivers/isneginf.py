import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    input_tensor = torch.tensor(input["input"])
    
    result_tensor = torch.isneginf(input_tensor)
    
    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"isneginf_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        
        neg_inf_constant = tf.constant(-np.inf)
        result_tensor = tf.math.equal(input_tensor, neg_inf_constant)

        return {"isneginf_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-np.inf, np.inf, 1.2], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert the results to numpy arrays and compare them
    torch_result_np = np.array(torch_result["isneginf_result"])
    tf_result_np = np.array(tf_result["isneginf_result"])

    if np.array_equal(torch_result_np, tf_result_np):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()