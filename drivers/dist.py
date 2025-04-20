import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])
    p = input.get("p", 2)

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    # Calculate the p-norm distance using torch.dist
    distance = torch.dist(input_tensor, other_tensor, p=p)
    
    if not cpu:
        distance = distance.cpu()
        
    return {"torch_dist": float(distance.item())}

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
        other_tensor = tf.constant(input["other"])
        p = input.get("p", 2)
        
        # Calculate the p-norm distance using TensorFlow
        tf_diff = input_tensor - other_tensor
        tf_norm = tf.norm(tf_diff, ord=p, axis=None)
        
        return {"tf_dist": float(tf_norm.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32), 
        "other": np.array([3.0, 2.0, 1.0], dtype=np.float32), 
        "p": 2
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality in common format
    assert np.isclose(torch_result["torch_dist"], tf_result["tf_dist"]), "Outputs are not equal"
    print("equal")

if __name__ == "__main__":
    main()