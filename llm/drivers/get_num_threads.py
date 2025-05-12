import numpy as np
import os

def torch_version(input_dict, cpu=True):
    import torch

    result = torch.get_num_threads()
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    result = tf.config.threading.get_intra_op_parallelism_threads()
    
    return {"result": np.array(result)}

def main():

    # Example input
    input_data = {}

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()