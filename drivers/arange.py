import numpy as np
from src.type_mapping_torch import np_to_torch

def torch_version(input, cpu=True):
    import torch
    
    start = input.get("start", 0)
    end = input["end"]
    step = input.get("step", 1)
    dtype = np_to_torch(input["dtype"])
    
    device = torch.device('cpu' if cpu else 'cuda')
    
    result = torch.arange(start=start, end=end, step=step, dtype=dtype, device=device)
    
    if not cpu:
        result = result.cpu()
    
    return {'output':result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    
    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        start = input.get("start", 0)
        end = input["end"]
        step = input.get("step", 1)
        dtype = tf.as_dtype(input["dtype"])
        
        result = tf.range(start=start, limit=end, delta=step, dtype=dtype)
    
    return {'output':result.numpy()}

def main():
    # Example input
    input_data = {
        "start": 0,
        "end": 5,
        "step": 1,
        "dtype": np.int64
    }
    
    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Assert and print equality
    if np.allclose(torch_result['output'], tf_result['output']):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()