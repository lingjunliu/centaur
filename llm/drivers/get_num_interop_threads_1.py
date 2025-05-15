import numpy as np
import os
import multiprocessing

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    result = torch.get_num_interop_threads()
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    try:
        result = tf.config.threading.get_inter_op_parallelism_threads()
        if result == 0:
            result = os.cpu_count()
    except:
        result = os.cpu_count()
    
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()