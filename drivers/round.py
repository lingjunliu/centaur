import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    decimals = input.get("decimals", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply torch.round
    if decimals != 0:
        factor = torch.tensor(10.0 ** decimals, device=input_tensor.device)
        rounded_tensor = torch.round(input_tensor * factor) / factor
    else:
        rounded_tensor = torch.round(input_tensor)
    
    if not cpu:
        rounded_tensor = rounded_tensor.cpu()
        
    return {"rounded_tensor": rounded_tensor.numpy()}

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
        decimals = input.get("decimals", 0)

        # Apply TensorFlow equivalent
        if decimals != 0:
            factor = tf.constant(10.0 ** decimals)
            rounded_tensor = tf.math.round(input_tensor * factor) / factor
        else:
            rounded_tensor = tf.math.round(input_tensor)
        
        return {"rounded_tensor": rounded_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.1234, 4.5678, -2.3456], dtype=np.float32),
        "decimals": 2
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert to check equality
    if np.allclose(torch_result["rounded_tensor"], tf_result["rounded_tensor"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()