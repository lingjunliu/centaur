import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply to torch.signbit
    output_tensor = torch.signbit(input_tensor)
    
    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"signbit_output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent operation
        zero_tensor = tf.constant(0.0)
        negative_input = tf.less(input_tensor, zero_tensor)
        negative_zero_input = tf.equal(input_tensor, -zero_tensor) & tf.equal(tf.math.sign(tf.bitcast(input_tensor, tf.int32)), -1)

        output_tensor = tf.logical_or(negative_input, negative_zero_input)
        
        return {"signbit_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.7, -1.2, 0.0, 2.3, -0.0], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result["signbit_output"], tf_result["signbit_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()