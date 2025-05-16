import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    if not cpu:
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    result = torch.seed()

    return {"result": np.array([result])}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        seed_val = 42
        tf.random.set_seed(seed_val)
        
        # Get a random integer in the range of int64
        rand_int = tf.random.uniform(shape=[], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64)
        result = rand_int.numpy()
        
    return {"result": np.array([result])}

def main():
    A_TOL = 2**62 # Set a large tolerance

    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()