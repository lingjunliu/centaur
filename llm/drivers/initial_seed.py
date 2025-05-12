import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    torch.manual_seed(42)
    result = torch.initial_seed()
    
    return {"result": int(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    tf.random.set_seed(42)
    result = tf.random.uniform(shape=(1,), minval=0, maxval=2**31-1, dtype=tf.int64).numpy()[0]

    return {"result": int(result)}

def main():
    A_TOL = 1.0  # Increased tolerance

    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert abs(torch_result["result"] - tf_result["result"]) < A_TOL

    print("Success")

if __name__ == "__main__":
    main()