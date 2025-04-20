import torch
import tensorflow as tf
import itertools
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    r = input.get("r", 2)
    with_replacement = input.get("with_replacement", False)

    # Apply to torch.combinations
    combinations = torch.combinations(input_tensor, r=r, with_replacement=with_replacement)
    
    if not cpu:
        combinations = combinations.cpu()

    return {"combinations": combinations.numpy()}

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
        r = input.get("r", 2)
        with_replacement = input.get("with_replacement", False)
        
        input_list = input["input"]
        
        # Generate combinations using itertools
        if with_replacement:
            combs = list(itertools.combinations_with_replacement(input_list, r))
        else:
            combs = list(itertools.combinations(input_list, r))
        
        combs_tensor = tf.constant(combs)

        return {"combinations": combs_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": [1, 2, 3],
        "r": 2,
        "with_replacement": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if they are equal
    assert np.array_equal(torch_result["combinations"], tf_result["combinations"]), "Results do not match"
    print("equal")

if __name__ == "__main__":
    main()