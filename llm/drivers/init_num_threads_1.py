import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    num_threads = input_dict["num_threads"]
    
    torch.set_num_threads(num_threads)

    return {"result": torch.get_num_threads()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import os

    num_threads = input_dict["num_threads"]
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(num_threads)

    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    
    return {"result": num_threads}

def main():
    A_TOL = 0.01
    input_data = {
        "num_threads": 4,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "num_threads": 8,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()